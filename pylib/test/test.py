import BASIC_changepoint
import numpy as np

def default_params(model):
    T = 1000
    J = 9
    params = {}
    if model == 'normal_mean':
        params['mu_0'] = 0.0
        params['sigma^2'] = 1.0
        params['lambda'] = 0.03
    elif model == 'normal_var':
        params['mu_0'] = 0.0
        params['alpha'] = 0.3
        params['beta'] = 1.0
    elif model == 'normal_mean_var':
        params['mu_0'] = 0.0
        params['lambda'] = 0.03
        params['alpha'] = 1.0
        params['beta'] = 1.0
    elif model == 'poisson':
        params['alpha'] = 2.0
        params['beta'] = 1.0
    elif model == 'bernoulli':
        params['alpha'] = 0.3
        params['beta'] = 0.3
    elif model == 'laplace_scale':
        params['alpha'] = 1.0
        params['beta'] = 1.0
    else:
        raise RuntimeError("Unrecognized model")
    q_vals = [0,0.1,0.2,0.3,0.4]
    pi_q = [0.99,0.0025,0.0025,0.0025,0.0025]
    return T, J, params, q_vals, pi_q

def generate_data(T, J, model, params, q_vals, pi_q):
    Z = np.zeros((J,T), dtype='bool')
    for t in range(1,T):
        q = q_vals[np.nonzero(np.random.multinomial(1, pi_q))[0][0]]
        for j in range(J):
            if np.random.random() < q:
                Z[j,t] = True
    if model in ['normal_mean', 'normal_var', 'normal_mean_var',
            'laplace_scale']:
        X = np.zeros((J,T), dtype='float')
    else:
        X = np.zeros((J,T), dtype='int')
    for j in range(J):
        start = 0
        for t in range(1,T+1):
            if t == T or Z[j,t]:
                end = t
                if model == 'normal_mean':
                    var = params['sigma^2']
                    mu = np.random.normal(params['mu_0'],
                            np.sqrt(var/params['lambda']))
                    X[j,start:end] = np.random.normal(mu,np.sqrt(var),end-start)
                elif model == 'normal_var':
                    var = 1.0/np.random.gamma(params['alpha'],
                            1.0/params['beta'])
                    mu = params['mu_0']
                    X[j,start:end] = np.random.normal(mu,np.sqrt(var),end-start)
                elif model == 'normal_mean_var':
                    var = 1.0/np.random.gamma(params['alpha'],
                            1.0/params['beta'])
                    mu = np.random.normal(params['mu_0'],
                            np.sqrt(var/params['lambda']))
                    X[j,start:end] = np.random.normal(mu,np.sqrt(var),end-start)
                elif model == 'poisson':
                    lamb = np.random.gamma(params['alpha'],1.0/params['beta'])
                    X[j,start:end] = np.random.poisson(lamb,end-start)
                elif model == 'bernoulli':
                    prob = np.random.beta(params['alpha'],params['beta'])
                    X[j,start:end] = (np.random.rand(end-start)<prob)
                elif model == 'laplace_scale':
                    scale = 1.0/np.random.gamma(params['alpha'],
                            1.0/params['beta'])
                    X[j,start:end] = np.random.laplace(0,scale,end-start)
                else:
                    raise RuntimeError("Unrecognized model")
                start = end
    return X, Z

def test_model(model):
    print(f"Testing {model}")
    T, J, params, q_vals, pi_q = default_params(model)
    X, Z = generate_data(T, J, model, params, q_vals, pi_q)
    true_chg_fig = BASIC_changepoint.plot_changes(X,
            BASIC_changepoint.Z_matrix_to_change_dict(Z))
    [samples, params_est, q_vals_est, pi_q_est] = \
            BASIC_changepoint.MCMC_sample(X, model)
    chg_probs = BASIC_changepoint.marginal_change_probs(samples, J, T)
    mode = BASIC_changepoint.compute_posterior_mode(X, model, params_est,
            q_vals_est, pi_q_est, Z=(chg_probs>0.5))
    detected_chg_fig = BASIC_changepoint.plot_changes(X, mode)
    true_chg_fig.suptitle('True changes for %s model' % model)
    detected_chg_fig.suptitle('Detected changes for %s model' % model)
    true_chg_fig.show()
    detected_chg_fig.show()

if __name__ == '__main__':
    np.random.seed(123)
    test_model('normal_mean')
    input(f"[PRESS ENTER TO CONTINUE]")
    test_model('normal_var')
    input(f"[PRESS ENTER TO CONTINUE]")
    test_model('normal_mean_var')
    input(f"[PRESS ENTER TO CONTINUE]")
    test_model('poisson')
    input(f"[PRESS ENTER TO CONTINUE]")
    test_model('bernoulli')
    input(f"[PRESS ENTER TO CONTINUE]")
    test_model('laplace_scale')
    input(f"[PRESS ENTER TO QUIT.")
