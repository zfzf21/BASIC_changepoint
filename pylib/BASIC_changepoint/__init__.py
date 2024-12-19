import numpy as np
import BASIC_changepoint._c_funcs as _c_funcs
from BASIC_changepoint._version import __version__

def seed(s):
    ''' Seed random number generator for MCMC sampler

    Inputs
    s: integer seed
    '''
    _c_funcs.seed(int(s))

def MCMC_sample(X, model, sample_iters=200, MCEM_schedule=[10,20,40,60,100],
        Z=None, model_params=None, pi_q=None, q_vals=None,
        estimate_pi_q=True, row_time_block=50, col_sample_approx=1e-6,
        col_only=False, row_only=False, nswaps=None, verbose=True):
    ''' Sample changepoints with MCMC and perform MCEM parameter estimation

    Example usage for a normal model with changing mean, to perform 200
    sampling iterations and MCEM prior parameter updates after iterations
    10, 20, 40, 60, and 100:

        [samples, model_params, pi_q, q_vals] = MCMC_sample(X,
                'normal_mean', sample_iters=200,
                MCEM_schedule=[10,20,40,60,100])

    Returns: [ samples, model_params, pi_q, q_vals ]
        samples: [ { int: set([ int, ..., int ]), ... },
                        ...
                   { int: set([ int, ..., int ]), ... } ]
            List of length sample_iters, containing the MCMC samples of the
            changepoints. Each sample is represented as a dictionary
            { int: set([ int, ..., int ]), ... } where each key is the
            sequential position/time point (in 0,...,T-1) of a change and
            the corresponding value is the set of indices (in 0,...,J-1) of
            sequences carrying a change at that position/time. (WARNING:
            Both sequential positions and sequence indices are 0-indexed.)
            To convert this sample into a boolean J x T matrix of
            changepoint indicators, use the change_dict_to_Z_matrix
            function.
        model_params: Final value of model_params (likelihood model
            conjugate prior parameters) as estimated by MCEM. Same as input 
            model_params if MCEM is not applied.
        pi_q: Support points of the changepoint frequency prior. Same as
            input pi_q if specified, otherwise returns the default
            initialization of pi_q.
        q_vals: Final value of q_vals (changepoint frequency prior
            probability weights) as estimated by MCEM. Same as input q_vals
            if MCEM is not applied.

    Inputs
    X: 2-D numpy array (float)
        Data to be analyzed, containing J rows (sequences) and T columns
        (time points / sequential positions).
    model: str
        Specifies the likelihood model. Available options:
            'normal_mean' -- Normal, changing mean, fixed variance
            'normal_var' -- Normal, changing variance, fixed mean
            'normal_mean_var' -- Normal, changing variance, changing mean
            'poisson' -- Poisson, changing mean
            'bernoulli' -- Bernoulli, changing success probability
            'laplace_scale' -- Laplace, fixed zero mean, changing scale
    sample_iters: int
        Number of MCMC sampling iterations.
    MCEM_schedule: [ int, ..., int ]
        Perform MCEM updates of the likelihood model conjugate prior
        parameters (model_params) and changepoint frequency prior
        probability masses (q_vals) after these numbers of total MCMC
        iterations. To keep the initial parameters throughout, set to the
        empty list [].
    Z: 2-D numpy array (bool) or None
        Initial state for the MCMC sampler. If specified, must be of the
        same shape as X, with first column 0 (False). If Z=None, sampler is
        initialized by default to the zero matrix.
    model_params: { str: float, ..., str: float } or None
        Initial likelihood model conjugate prior parameters. (Initialized
        empirically from the data if model_params=None.) If specified, must
        be in the following model-dependent format:
            normal_mean --
                {'mu_0':float, 'sigma^2':float, 'lambda':float} 
                variance fixed at sigma^2
                mean ~ Normal(mu_0, sigma^2/lambda)
            normal_var --
                {'mu_0':float, 'alpha':float, 'beta':float}
                mean fixed at mu_0
                variance ~ InverseGamma(alpha, beta)
            normal_mean_var --
                {'mu_0':float, 'lambda':float, 'alpha':float, 'beta':float}
                variance ~ InverseGamma(alpha, beta)
                mean ~ Normal(mu_0, variance/lambda)
            poisson -- 
                {'alpha':float, 'beta':float}
                mean ~ Gamma(alpha, beta)
            bernoulli --
                {'alpha':float, 'beta':float}
                success probability ~ Beta(alpha, beta)
            laplace_scale --
                {'alpha':float, 'beta':float}
                mean fixed at 0
                scale ~ InverseGamma(alpha, beta)
    pi_q: [ float, ..., float ] or None
        Support points for the changepoint frequency prior. If specified,
        must take values in [0,1] (with recommended values in [0,1/2)). If
        pi_q=None, support points initialized to 0, 1/J, 2/J, ...,
        floor((J-1)/2)/J.
    q_vals: [ float, ..., float ] or None
        Initial probability masses for the changepoint frequency prior. If
        specified, must be of the same length as pi_q, nonnegative, and sum 
        to 1. If q_vals=None, probability mass at 0 is initialized to 0.9
        and the remaining mass of 0.1 is equally distributed across the
        other support points.
    estimate_pi_q: bool
        Set to True to update both model_params and q_vals during MCEM,
        False to update only model_params.
    row_time_block: int
        Divide exact row sampling into blocks of this size. Set to -1 to
        perform exact sampling of the entire rows (runtime may be slow for
        large data sets).
    col_sample_approx: float
        Error tolerance for column sampling approximation. See paper (Fan
        and Mackey) for details. Set to -1 to perform exact computations
        during column sampling (runtime may be slow for large data sets).
    col_only: bool
        If True, row-wise sampling of Z is not performed.
    row_only: bool
        If True, column-wise sampling of Z is not performed. (Default False,
        can set to True to improve per-iteration runtime for large data
        sets.)
    nswaps: int or None
        Number of Metropolis-Hastings column swaps of Z to perform per MCMC
        iteration. If nswaps=None, set to default value of 10*T where T is
        the number of columns of X.
    verbose: bool
        Print sampler progress and MCEM optimization output to screen.
    '''
    if np.isnan(X).any() or np.isinf(X).any():
        raise RuntimeError("Input data has missing or infinite values")
    if q_vals is not None or pi_q is not None:
        if q_vals is None or pi_q is None:
            raise RuntimeError("Must specify both q_vals and pi_q")
        q_vals = list(q_vals)
        pi_q = list(pi_q)
        if len(q_vals) != len(pi_q):
            raise RuntimeError("Incompatible q_vals and pi_q")
    elif len(MCEM_schedule) == 0:
        print("WARNING: Default priors used with no MCEM updates")
    if col_sample_approx > 0 and q_vals is not None:
        print("WARNING: col_sample_approx > 0 has not been tested with non-default q_vals")
    if Z is None:
        Z = np.zeros(X.shape, dtype='bool')
    else:
        Z = Z.copy()
    if nswaps is None:
        nswaps = -1
    MCEM_schedule = list(MCEM_schedule)
    # Calls py_gibbs_sample in py_extension.cpp, which wraps gibbs_sample in
    # inference_procedures.cpp
    return _c_funcs.gibbs_sample(X,Z,model,model_params,q_vals,pi_q,
            sample_iters,MCEM_schedule,estimate_pi_q,row_time_block,
            col_sample_approx,col_only,row_only,nswaps,verbose)

def compute_posterior_mode(X, model, model_params, pi_q, q_vals, Z=None,
        max_iters=100, row_time_block=50, col_only=False, row_only=False,
        swap=True, verbose=True):
    ''' Estimate changepoints via posterior maximization

    Prior parameters must be specified; these may be estimated via MCEM
    using the MCMC_sample routine. An initial guess for Z may be provided.
    Example usage:

        [samples, model_params, pi_q, q_vals] = MCMC_sample(X,
                'normal_mean', sample_iters=200,
                MCEM_schedule=[10,20,40,60,100])
        chg_probs = marginal_change_probs(samples[100:], X.shape[0],
                X.shape[1])
        mode = compute_posterior_mode(X, 'normal_mean', model_params, pi_q,
                q_vals, Z=(chg_probs>0.5))

    The maximization procedure is iterative and local, and depends on the
    initial guess for Z.

    Returns: mode
        The maximum a posteriori estimate of changepoints in all sequences,
        represented as a dictionary { int: set([ int, ..., int ]), ... }
        where each key is the sequential position/time point (in 0,...,T-1)
        of a change and the corresponding value is the set of indices
        (in 0,...,J-1) of sequences carrying a change at that
        position/time. (WARNING: Both sequential positions and sequence
        indices are 0-indexed.) To convert this sample into a boolean
        J x T matrix of changepoint indicators, use the
        change_dict_to_Z_matrix function.

    Inputs
    X: 2-D numpy array (float)
        Data to be analyzed, containing J rows (sequences) and T columns
        (time points / sequential positions).
    model: str
        Specifies the likelihood model. Available options:
            'normal_mean' -- Normal, changing mean, fixed variance
            'normal_var' -- Normal, changing variance, fixed mean
            'normal_mean_var' -- Normal, changing variance, changing mean
            'poisson' -- Poisson, changing mean
            'bernoulli' -- Bernoulli, changing success probability
            'laplace_scale' -- Laplace, fixed zero mean, changing scale
    model_params: { str: float, ..., str: float } or None
        Likelihood model conjugate prior parameters. Must be in the
        following model-dependent format:
            normal_mean --
                {'mu_0':float, 'sigma^2':float, 'lambda':float} 
                variance fixed at sigma^2
                mean ~ Normal(mu_0, sigma^2/lambda)
            normal_var --
                {'mu_0':float, 'alpha':float, 'beta':float}
                mean fixed at mu_0
                variance ~ InverseGamma(alpha, beta)
            normal_mean_var --
                {'mu_0':float, 'lambda':float, 'alpha':float, 'beta':float}
                variance ~ InverseGamma(alpha, beta)
                mean ~ Normal(mu_0, variance/lambda)
            poisson -- 
                {'alpha':float, 'beta':float}
                mean ~ Gamma(alpha, beta)
            bernoulli --
                {'alpha':float, 'beta':float}
                success probability ~ Beta(alpha, beta)
            laplace_scale --
                {'alpha':float, 'beta':float}
                mean fixed at 0
                scale ~ InverseGamma(alpha, beta)
    pi_q: [ float, ..., float ]
        Support points for the changepoint frequency prior. Must take
        values in [0,1] (with recommended values in [0,1/2)).
    q_vals: [ float, ..., float ]
        Probability masses for the changepoint frequency prior. Must be of
        the same length as pi_q, nonnegative, and sum to 1.
    Z: 2-D numpy array (bool) or None
        Initial guess for the posterior mode. If specified, must be of the
        same shape as X, with first column 0 (False). If Z=None, algorithm
        is initialized at the zero matrix.
    max_iters: int
        Maximum number of iterations to perform.
    row_time_block: int
        Divide exact row maximization into blocks of this size. Set to -1 to
        perform exact maximization over entire rows (runtime may be slow for
        large data sets).
    col_only: bool
        If True, row-wise maximization of Z is not performed.
    row_only: bool
        If True, column-wise maximization of Z is not performed.
    swap: bool
        If True, perform column swapping subroutine.
    verbose: bool
        Print iteration progress to screen.
    '''
    if np.isnan(X).any() or np.isinf(X).any():
        raise RuntimeError("Input data has missing or infinite values")
    q_vals = list(q_vals)
    pi_q = list(pi_q)
    if len(q_vals) != len(pi_q):
        raise RuntimeError("Incompatible q_vals and pi_q")
    if Z is None:
        Z = np.zeros(X.shape, dtype='bool')
    else:
        Z = Z.copy()
    # Calls py_compute_posterior_mode in py_extension.cpp, which wraps
    # compute_posterior_mode in inference_procedures.cpp
    _c_funcs.compute_posterior_mode(X,Z,model,model_params,q_vals,pi_q,
            max_iters,row_time_block,col_only,row_only,swap,verbose)
    chg = Z_matrix_to_change_dict(Z)
    return chg

def Z_matrix_to_change_dict(Z):
    ''' Convert a boolean changepoint matrix to a dictionary representation

    Returns: chg_dict
        chg_dict: { int: set([ int, ..., int ]), ... } 
            Each key is the sequential position/time point (in 0,...,T-1) of
            a change and the corresponding value is the set of indices
            (in 0,...,J-1) of sequences carrying a change at that
            position/time. (WARNING: Both sequential positions and sequence
            indices are 0-indexed.) 

    Inputs
    Z: 2-D numpy array (bool)
        Boolean changepoint matrix, containing J rows (sequences) and T
        columns (time points / sequential positions). Entry (j,t) is True
        if a changepoint occurs at position t in sequence j.
    '''
    chg = {}
    for t in range(Z.shape[1]):
        for j in range(Z.shape[0]):
            if Z[j,t]:
                if t not in chg:
                    chg[t] = []
                chg[t] += [j]
    return chg

def change_dict_to_Z_matrix(chg,J,T):
    ''' Convert a dict representation of changepoints to a boolean matrix

    Returns: Z
        Z: 2-D numpy array (bool)
            Boolean changepoint matrix, containing J rows (sequences) and T
            columns (time points / sequential positions). Entry (j,t) is
            True if a changepoint occurs at position t in sequence j.

    Inputs
    chg: { int: set([ int, ..., int ]), ... } 
        Each key is the sequential position/time point (in 0,...,T-1) of a
        change and the corresponding value is the set of indices
        (in 0,...,J-1) of sequences carrying a change at that
        position/time. (WARNING: Both sequential positions and sequence
        indices are 0-indexed.)
    J: int
        Total number of sequences
    T: int
        Total number of sequential positions/time points
    '''
    Z = np.zeros((J,T), dtype='bool')
    for t,inds in chg.items():
        Z[inds,t] = True
    return Z

def marginal_change_probs(samples,J,T):
    ''' Compute marginal posterior change probabilities from MCMC samples

    Returns: chg_probs
        chg_probs: 2-D numpy array (float)
            Matrix of J rows (sequences) and T columns (time points /
            sequential positions). Entry (j,t) is the marginal probability
            of a changepoint occuring at position t in sequence j, under the
            posterior distribution as estimated from the input MCMC samples.

    Inputs
    samples: [ { int: set([ int, ..., int ]), ... },
                    ...
               { int: set([ int, ..., int ]), ... } ]
        Changepoints sampled from the posterior distribution. See
        documentation of MCMC_sample for formatting details.
    J: int
        Total number of sequences
    T: int
        Total number of sequential positions/time points
    '''
    chg_probs = np.zeros((J,T), dtype='float')
    for chg in samples:
        for t,inds in chg.items():
            chg_probs[inds,t] += 1
    chg_probs /= len(samples)
    return chg_probs

def plot_changes(X, chg, seqs=[], t_range=[]):
    ''' Plot changepoint locations together with sequence data

    Returns: fig
        A matplotlib Figure instance; it may be displayed using fig.show().

    Inputs
    X: 2-D numpy array (float)
        Data matrix containing J rows (sequences) and T columns (time
        points / sequential positions).
    chg: { int: set([ int, ..., int ]), ... } 
        A set of changepoints in X, where each key is the sequential
        position/time point (in 0,...,T-1) of a change and the corresponding
        value is the set of indices (in 0,...,J-1) of sequences carrying a
        change at that position/time. (E.g., this may be a changepoint set
        corresponding to one sample from MCMC_sample, or the posterior mode
        estimate from compute_posterior_mode. Position and sequence indices
        must correspond correctly with rows and columns of X.)
    seqs: [ int, ..., int ]
        List of sequences of X to plot. (WARNING: Sequences of X are
        0-indexed.) This function can plot at most 9 sequences
        simultaneously; if seqs=[], then the first 9 sequences in X are
        plotted by default.
    t_range: [ int, int ]
        Range of sequential positions/time points to plot, specified as
        [t_min, t_max+1]. If t_range=[], then this defaults to the entire
        sequence ([0,T] where T is the total number of columns of X).
    '''
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    if len(seqs) > 9:
        seqs = seqs[:9]
    if len(seqs) == 0:
        seqs = range(9)
    if t_range == []:
        t_inds = range(X.shape[1])
    else:
        t_inds = range(t_range[0],t_range[1])
    Z = change_dict_to_Z_matrix(chg, X.shape[0], X.shape[1])
    fig = plt.figure()
    gs = gridspec.GridSpec(len(seqs),1, wspace=0, hspace=0.2)
    for j in range(len(seqs)):
        if j == 0:
            ax = fig.add_subplot(gs[j])
            ax0 = ax
        else:
            ax = fig.add_subplot(gs[j], sharex=ax0)
        ax.plot(t_inds, X[seqs[j],t_inds], 'k.')
        if j != len(seqs)-1:
            ax.xaxis.set_visible(False)
        ax.set_xlim(min(t_inds), max(t_inds))
        ymax = max(X[seqs[j],t_inds])
        ymin = min(X[seqs[j],t_inds])
        ax.set_ylim([ymin-(ymax-ymin)*0.2,ymax+(ymax-ymin)*0.2])
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        if len(np.nonzero(Z[seqs[j],:])[0]) > 0:
            ax.vlines(np.nonzero(Z[seqs[j],:])[0] - 0.5, ymin-(ymax-ymin)*0.2,
                    ymax+(ymax-ymin)*0.2, 'k', linewidth=2)
    return fig
