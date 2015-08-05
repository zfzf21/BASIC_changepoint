#include "inference_procedures.hpp"
#include "base.hpp"
#include "bernoulli_model.hpp"
#include "laplace_scale_model.hpp"
#include "normal_mean_model.hpp"
#include "normal_mean_var_model.hpp"
#include "normal_var_model.hpp"
#include "poisson_model.hpp"
#include "dlib/optimization.h"
#include <algorithm>
#include <list>
#include <random>
#include <time.h>

using namespace bayesian_simultaneous_changepoint;

/* Random number generators */
static std::mt19937 GEN;
static std::uniform_real_distribution<double> UNIF(0.0, 1.0);
void bayesian_simultaneous_changepoint::set_seed(int s) {
    GEN = std::mt19937(s);
}

/* Default initialization for changepoint prior */
ChangeParams* init_change_params(int J) {
    ChangeParams* chg_params = new ChangeParams();
    if (J <= 9)
        J = 10;
    chg_params->Q = J/2;
    chg_params->q_vals = new double[J/2];
    chg_params->log_piq = new double[J/2];
    for (int j = 0; j < J/2; ++j) {
        chg_params->q_vals[j] = double(j)/J;
        if (j == 0)
            chg_params->log_piq[j] = log(0.9);
        else
            chg_params->log_piq[j] = log(0.1/(J/2-1));
    }
    return chg_params;
}
/* Check prior parameters */
template <typename Model>
void check_model_params(const ModelParams* params) {
    for (int k = 0; k < params->K; ++k) {
        if (params->vals[k] < Model::lower_bound(params->names[k])) {
            std::stringstream ss;
            ss << "Parameter " << params->names[k] << " with value "
                << params->vals[k] << " is less than lower bound of "
                << Model::lower_bound(params->names[k]);
            throw std::runtime_error(ss.str());
        }
        if (params->vals[k] > Model::upper_bound(params->names[k])) {
            std::stringstream ss;
            ss << "Parameter " << params->names[k] << " with value "
                << params->vals[k] << " is greater than upper bound of "
                << Model::upper_bound(params->names[k]);
            throw std::runtime_error(ss.str());
        }
    }
}
void check_change_params(const ChangeParams* chg_params) {
    double log_piq_sum = NEGINF;
    for (int i = 0; i < chg_params->Q; ++i) {
        if (chg_params->q_vals[i] < 0 || chg_params->q_vals[i] > 1)
            throw std::runtime_error(
                    "q_val values must be between 0 and 1");
        log_piq_sum = logaddexp(log_piq_sum, chg_params->log_piq[i]);
    }
    if (fabs(log_piq_sum) > 1e-6)
        throw std::runtime_error("pi_q values must sum to 1");
}

/**************** GIBBS SAMPLING AND MCEM PROCEDURES ****************/
template <typename Model>
void sample_rows(const double* X,
        bool* Z, int T, int J, const ModelParams& params,
        const ChangeParams& chg_params, int first_block=-1, int block=-1) {
    /* Count number of changepoints at each time */
    int* N = new int[T];
    for (int t = 1; t < T; ++t) {
        N[t] = 0;
        for (int j = 0; j < J; ++j)
            N[t] += (int)Z[j*T+t];
    }
    double* logc = new double[T];
    double* logQ = new double[T];
    Model prev_model(params);
    Model next_model(params);
    Model model(params);
    for (int j = 0; j < J; ++j) {
        prev_model.clear();
        next_model.clear();
        model.clear();
        for (int t = 1; t < T; ++t) {
            N[t] -= (int)Z[j*T+t];
            logc[t] = chg_params.log_f[N[t]+1]-chg_params.log_g[N[t]+1];
        }
        int start = 1;
        int end = (first_block > 0 && first_block < T ? first_block : T);
        int next = end;
        for (int t = end; t < T; ++t) {
            next = t;
            if (Z[j*T+t]) break;
            next_model.add_data(X[j*T+t]);
        }
        while (start < T) {
            for (int t = end-1; t >= start-1; --t) {
                model.clear();
                if (t == start-1)
                    model.combine(prev_model);
                double log_nochg_prob = 0;
                logQ[t] = NEGINF;
                for (int s = t+1; s < end; ++s) {
                    model.add_data(X[j*T+s-1]);
                    double val = model.logmarginal()+logQ[s]+log_nochg_prob
                        + logc[s];
                    logQ[t] = logaddexp(logQ[t],val);
                    log_nochg_prob += logz(1.0-exp(logc[s]));
                }
                model.add_data(X[j*T+end-1]);
                model.combine(next_model);
                logQ[t] = logaddexp(logQ[t],
                        model.logmarginal()+log_nochg_prob);
            }
            model.clear();
            model.combine(prev_model);
            int s = start-1;
            double rand_unif = UNIF(GEN);
            double log_nochg_prob = 0;
            for (int t = start; t < end; ++t) {
                model.add_data(X[j*T+t-1]);
                prev_model.add_data(X[j*T+t-1]);
                if (rand_unif < exp(model.logmarginal()+logQ[t]
                            +log_nochg_prob+logc[t]-logQ[s])) {
                    Z[j*T+t] = true;
                    ++N[t];
                    s = t;
                    rand_unif = UNIF(GEN);
                    log_nochg_prob = 0;
                    model.clear();
                    prev_model.clear();
                } else {
                    Z[j*T+t] = false;
                    rand_unif -= exp(model.logmarginal()+logQ[t]
                            +log_nochg_prob+logc[t]-logQ[s]);
                    log_nochg_prob += logz(1.0-exp(logc[t]));
                }
            }
            start = end;
            if (block > 0 && end + block > next) {
                end = (end + block < T ? end + block : T);
                next_model.clear();
                for (int t = end; t < T; ++t) {
                    next = t;
                    if (Z[j*T+t]) break;
                    next_model.add_data(X[j*T+t]);
                }
            } else if (block > 0) {
                for (int t = end; t < end + block; ++t)
                    next_model.remove_data(X[j*T+t]);
                end += block;
            }
        }
    }
    delete[] N;
    delete[] logc;
    delete[] logQ;
}

template <typename Model>
void sample_columns(const double* X,
        bool* Z, int T, int J, const ModelParams& params,
        const ChangeParams& chg_params, double approx=-1) {
    /* Set up three models to compute likelihoods for each observable */
    Model** model_before = new Model*[J];
    Model** model_after = new Model*[J];
    Model** model_all = new Model*[J];
    for (int j = 0; j < J; ++j) {
        model_before[j] = new Model(params);
        model_after[j] = new Model(params);
        model_all[j] = new Model(params);
        model_before[j]->add_data(X[j*T]);
        model_all[j]->add_data(X[j*T]);
        for (int t = 1; t < T; ++t) {
            if (Z[j*T+t])
                break;
            model_all[j]->add_data(X[j*T+t]);
            model_after[j]->add_data(X[j*T+t]);
        }
    }
    /* logR[j*(J+1)+k] is log of coefficient of x^ky^(J-j-k) in expansion
     * of prod_{i=j}^{J-1} (A(i)x+B(i)y) where A(i) is likelihood of model
     * with change and B(i) is likelihood of model without change.
     * Here 0 <= k <= J-j, and logR[J*(J+1)+0] = 0. logR will be evaluated
     * lazily for k = 0, k = 1, k = 2, ... */
    double* logR = new double[(J+1)*(J+1)];
    for (int t = 1; t < T; ++t) {
        /* Update models if there is a change */
        for (int j = 0; j < J; ++j) {
            if (Z[j*T+t]) {
                model_after[j]->add_data(X[j*T+t]);
                model_all[j]->add_data(X[j*T+t]);
                for (int s = t+1; s < T; ++s) {
                    if (Z[j*T+s])
                        break;
                    model_after[j]->add_data(X[j*T+s]);
                    model_all[j]->add_data(X[j*T+s]);
                }
            }
        }
        /* Evaluate logR for k = 0 and all j */
        logR[J*(J+1)] = 0;
        for (int j = J-1; j >= 0; --j) {
            logR[j*(J+1)] = logR[(j+1)*(J+1)]+model_all[j]->logmarginal();
        }
        int max_k = 0;
        int N = 0;
        for (int j = 0; j < J; ++j) {
            double log_num = NEGINF;
            double log_denom = NEGINF;
            /* Compute sum over k in both numerator and denominator */
            for (int k = 0; k < J-j; ++k) {
                if (k > max_k) {
                    /* Evaluate logR for this k and all j */
                    max_k = k;
                    logR[(J-k)*(J+1)+k] = logR[(J-k+1)*(J+1)+k-1]
                        +model_before[J-k]->logmarginal()
                        +model_after[J-k]->logmarginal();
                    for (int i = J-k-1; i >= 0; --i) {
                        logR[i*(J+1)+k] = logaddexp(logR[(i+1)*(J+1)+k]
                                +model_all[i]->logmarginal(),
                                logR[(i+1)*(J+1)+k-1]
                                +model_before[i]->logmarginal()
                                +model_after[i]->logmarginal());
                    }
                }
                double log_denom_incr = logR[(j+1)*(J+1)+k]
                    + chg_params.log_g[N+k+1];
                double log_num_incr = logR[(j+1)*(J+1)+k]
                    + chg_params.log_f[N+k+1];
                log_denom = logaddexp(log_denom, log_denom_incr);
                log_num = logaddexp(log_num, log_num_incr);
                /* Terminate sum over k if increment is negligible*/
                if (approx > 0 && k > 0
                        && exp(log_denom_incr - log_denom) < approx
                        && exp(log_num_incr - log_num) < approx) {
                    break;
                }
            } 
            /* Sample change and update models */
            double log_expected_q = log_num-log_denom;
            double log_odds = model_before[j]->logmarginal()
                +model_after[j]->logmarginal()+log_expected_q
                -model_all[j]->logmarginal()-logz(1.0-exp(log_expected_q));
            if (log(UNIF(GEN)) < log_odds - logaddexp(0.0,log_odds)) {
                Z[j*T+t] = true;
                N += 1;
                model_before[j]->clear();
                model_all[j]->clear();
                model_all[j]->combine(*model_after[j]);
            } else
                Z[j*T+t] = false;
            model_before[j]->add_data(X[j*T+t]);
            model_after[j]->remove_data(X[j*T+t]);
        }
    }
    for (int j = 0; j < J; ++j) {
        delete[] model_before[j];
        delete[] model_after[j];
        delete[] model_all[j];
    }
    delete[] logR;
    delete[] model_before;
    delete[] model_after;
    delete[] model_all;
}

/* Helper functions used in metropolis_swap_change_times and
 * greedy_swap_change_times */
template<typename Model>
double evaluate_swap(int t, int t_new, const double* X, const bool* Z,
        int T, int J, Model** model_ptrs) {
    double log_old_prob = 0;
    double log_new_prob = 0;
    for (int j = 0; j < J; ++j) {
        if (Z[j*T+t] && Z[j*T+t_new]) continue;
        if (!Z[j*T+t] && !Z[j*T+t_new]) continue;
        if ((Z[j*T+t] && t_new == t+1)
                || (Z[j*T+t_new] && t == t_new+1)) {
            int t1 = (t > t_new ? t_new : t);
            log_old_prob += model_ptrs[j*T+t1-1]->logmarginal();
            log_old_prob += model_ptrs[j*T+t1]->logmarginal();
            model_ptrs[j*T+t1-1]->add_data(X[j*T+t1]);
            model_ptrs[j*T+t1]->remove_data(X[j*T+t1]);
            log_new_prob += model_ptrs[j*T+t1-1]->logmarginal();
            log_new_prob += model_ptrs[j*T+t1]->logmarginal();
            model_ptrs[j*T+t1]->add_data(X[j*T+t1]);
            model_ptrs[j*T+t1-1]->remove_data(X[j*T+t1]);
        } else if ((Z[j*T+t] && t_new == t-1)
                || (Z[j*T+t_new] && t == t_new-1)) {
            int t2 = (t > t_new ? t : t_new);
            log_old_prob += model_ptrs[j*T+t2-1]->logmarginal();
            log_old_prob += model_ptrs[j*T+t2]->logmarginal();
            model_ptrs[j*T+t2-1]->remove_data(X[j*T+t2-1]);
            model_ptrs[j*T+t2]->add_data(X[j*T+t2-1]);
            log_new_prob += model_ptrs[j*T+t2-1]->logmarginal();
            log_new_prob += model_ptrs[j*T+t2]->logmarginal();
            model_ptrs[j*T+t2]->remove_data(X[j*T+t2-1]);
            model_ptrs[j*T+t2-1]->add_data(X[j*T+t2-1]);
        } else {
            throw std::runtime_error("BUG -- Unhandled case");
        }
    }
    return exp(log_new_prob-log_old_prob);
}
template<typename Model>
void perform_swap(int t, int t_new, const double* X, bool* Z, int T, int J,
        Model** model_ptrs, int** chg_ptrs) {
    *chg_ptrs[t] = t_new;
    if (chg_ptrs[t_new])
        *chg_ptrs[t_new] = t;
    int* tmp = chg_ptrs[t_new];
    chg_ptrs[t_new] = chg_ptrs[t];
    chg_ptrs[t] = tmp;
    for (int j = 0; j < J; ++j) {
        if (Z[j*T+t] && Z[j*T+t_new]) continue;
        if (!Z[j*T+t] && !Z[j*T+t_new]) continue;
        if ((Z[j*T+t] && t_new == t+1)
                || (Z[j*T+t_new] && t == t_new+1)) {
            int t1 = (t > t_new ? t_new : t);
            Z[j*T+t1] = false;
            Z[j*T+t1+1] = true;
            model_ptrs[j*T+t1-1]->add_data(X[j*T+t1]);
            model_ptrs[j*T+t1]->remove_data(X[j*T+t1]);
            model_ptrs[j*T+t1] = model_ptrs[j*T+t1-1];
        } else if ((Z[j*T+t] && t_new == t-1)
                || (Z[j*T+t_new] && t == t_new-1)) {
            int t2 = (t > t_new ? t : t_new);
            Z[j*T+t2] = false;
            Z[j*T+t2-1] = true;
            model_ptrs[j*T+t2-1]->remove_data(X[j*T+t2-1]);
            model_ptrs[j*T+t2]->add_data(X[j*T+t2-1]);
            model_ptrs[j*T+t2-1] = model_ptrs[j*T+t2];
        } else {
            throw std::runtime_error("BUG -- Unhandled case");
        }
    }
}

template <typename Model>
void metropolis_swap_change_times(
        const double* X, bool* Z, int T, int J, const ModelParams& params,
        int nswaps) {
    Model** models = new Model*[T*J];
    Model** model_ptrs = new Model*[T*J];
    int* chg_times = new int[T];
    int** chg_ptrs = new int*[T];
    for (int t = 0; t < T; ++t)
        chg_ptrs[t] = NULL;
    int model_idx = 0;
    models[0] = new Model(params);
    int chg_idx = 0;
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; ++t) {
            if (Z[j*T+t]) {
                if (t == 0) {
                    throw std::runtime_error(
                            "Cannot have change at time 0");
                }
                ++model_idx;
                models[model_idx] = new Model(params);
                if (!chg_ptrs[t]) {
                    chg_times[chg_idx] = t;
                    chg_ptrs[t] = &(chg_times[chg_idx]);
                    ++chg_idx;
                }
            }
            models[model_idx]->add_data(X[j*T+t]);
            model_ptrs[j*T+t] = models[model_idx];
        }
        ++model_idx;
        models[model_idx] = new Model(params);
    }
    if (chg_idx == 0 || T < 3) {
        for (int i = 0; i <= model_idx; ++i)
            delete models[i];
        delete[] models;
        delete[] model_ptrs;
        delete[] chg_times;
        delete[] chg_ptrs;
        return;
    }
    for (int b = 0; b < nswaps; ++b) {
        /* Pick random change time and direction to move */
        int t = chg_times[(int)(UNIF(GEN) * chg_idx)];
        int t_new = (UNIF(GEN) < 0.5 ? t+1 : t-1);
        if (t_new == T) t_new = T-2;
        else if (t_new == 0) t_new = 2;
        double p = evaluate_swap<Model>(t,t_new,X,Z,T,J,model_ptrs);
        if (!chg_ptrs[t_new] && (t == 1 || t == T-1)) {
            p /= 2;
        } else if (!chg_ptrs[t_new] && (t_new == 1 || t_new == T-1)) {
            p *= 2;
        }
        if (p > 1 || UNIF(GEN) < p)
            perform_swap<Model>(t,t_new,X,Z,T,J,model_ptrs,chg_ptrs);
    }
    for (int i = 0; i <= model_idx; ++i)
        delete models[i];
    delete[] models;
    delete[] model_ptrs;
    delete[] chg_times;
    delete[] chg_ptrs;
}

typedef dlib::matrix<double,0,1> dlib_vector;
template <typename Model>
class model_params_objective {
    /* WARNING: These point to objects that might only exist in local scope
     * of instantiation */
    ModelParams* _params;
    std::vector<Model>* _models;
    public:
    model_params_objective(ModelParams& params, std::vector<Model>& models)
        : _params(&params), _models(&models) { }
    double operator() (const dlib_vector& x) const {
        for (int k = 0; k < _params->K; ++k)
            _params->vals[k] = x(k);
        double loglik = 0;
        for (typename std::vector<Model>::iterator iter =
                _models->begin(); iter != _models->end(); ++iter) {
            iter->reset_params(*_params);
            loglik += iter->logmarginal();
        }
        return loglik;
    }
};

template <typename Model>
void estimate_parameters(const double* X,
        int T, int J, const ChangeHist& chg_hist, unsigned MCEM_start,
        ModelParams& params, ChangeParams& chg_params,
        bool update_piq=true) {
    /* Compute sufficient statistics from data and change history to
     * perform MCEM */
    std::vector<Model> models;
    int* counts = new int[J+1];
    for (int j = 0; j < J+1; ++j)
        counts[j] = 0;
    int nchgs = 0;
    int* starts = new int[J];
    for (unsigned pos = MCEM_start; pos < chg_hist.size(); ++pos) {
        for (int j = 0; j < J; ++j)
            starts[j] = 0;
        for (Changes::const_iterator miter = chg_hist[pos].begin();
                miter != chg_hist[pos].end(); ++miter) {
            ++nchgs;
            ++counts[miter->second.size()];
            int end = miter->first;
            for (std::vector<int>::const_iterator jiter
                    = miter->second.begin(); jiter != miter->second.end();
                    ++jiter) {
                models.push_back(Model(params));
                for (int t = starts[*jiter]; t < end; ++t)
                    models.back().add_data(X[(*jiter)*T+t]);
                starts[*jiter] = end;
            }
        }
        for (int j = 0; j < J; ++j) {
            models.push_back(Model(params));
            for (int t = starts[j]; t < T; ++t)
                models.back().add_data(X[j*T+t]);
        }
    }
    int N = (T-1)*(chg_hist.size()-MCEM_start);
    counts[0] = N-nchgs;
    delete[] starts;

    /* Update model parameters using dlib::find_max_bobyqa */
    dlib_vector x(params.K);
    dlib_vector lower_bounds(params.K);
    dlib_vector upper_bounds(params.K);
    for (int k = 0; k < params.K; ++k) {
        x(k) = params.vals[k];
        lower_bounds(k) = Model::lower_bound(params.names[k]);
        upper_bounds(k) = Model::upper_bound(params.names[k]);
    }
    try {
        dlib::find_max_bobyqa(model_params_objective<Model>(params,models),
            x, 2*params.K+1, lower_bounds, upper_bounds, 10, 1e-6, 5000);
        for (int k = 0; k < params.K; ++k)
            params.vals[k] = x(k);
    } catch (std::exception& e) {
        std::cerr << "WARNING: BOBYQA optimization for model parameters "
            "did not succeed" << std::endl;
    }

    /* Update changepoint parameters */
    if (update_piq) {
        int iter;
        int MAX_ITER = 1000;
        double* log_denoms = new double[J+1];
        for (iter = 0; iter < MAX_ITER; ++iter) {
            double max_log_ratio = NEGINF;
            for (int j = 0; j <= J; ++j) {
                log_denoms[j] = NEGINF;
                for (int i = 0; i < chg_params.Q; ++i) {
                    log_denoms[j] = logaddexp(log_denoms[j],
                            chg_params.log_piq[i]
                            + j * logz(chg_params.q_vals[i])
                            + (J-j) * logz(1.0-chg_params.q_vals[i]));
                }
            }
            for (int i = 0; i < chg_params.Q; ++i) {
                double log_val = NEGINF;
                for (int j = 0; j <= J; ++j) {
                    if (counts[j] == 0)
                        continue;
                    log_val = logaddexp(log_val,
                            log(((double)counts[j]) / N)
                            + j * logz(chg_params.q_vals[i])
                            + (J - j) * logz(1-chg_params.q_vals[i])
                            - log_denoms[j]);
                }
                chg_params.log_piq[i] += log_val;
                max_log_ratio = (max_log_ratio > fabs(log_val)
                        ? max_log_ratio : fabs(log_val));
            }
            if (max_log_ratio < 1e-8)
                break;
        }
        delete[] log_denoms;
    }
    delete[] counts;
}

template <typename Model>
void bayesian_simultaneous_changepoint::gibbs_sample(const double* X,
        bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, ChangeHist& chg_hist,
        int sample_iters, const std::vector<int>& MCEM_schedule,
        bool estimate_piq, int row_sample_time_block,
        double col_sample_approx, bool col_only, bool row_only, int nswaps,
        bool verbose) {
    if (nswaps < 0)
        nswaps = 10*T;
    unsigned MCEM_pos = 0;
    if (!params) {
        params = Model::init_params(X, T, J);
    }
    if (!chg_params) {
        chg_params = init_change_params(J);
    }
    check_model_params<Model>(params);
    check_change_params(chg_params);
    if (verbose) {
        std::cout << "Initial model params:" << std::endl;
        std::cout << "  pi_q:" << std::endl;
        for (int i = 0; i < chg_params->Q; ++i)
            std::cout << "    " << chg_params->q_vals[i]
                << ": " << exp(chg_params->log_piq[i])
                << std::endl;
        for (int k = 0; k < params->K; ++k)
            std::cout << "  " << params->names[k] << ": "
                << params->vals[k] << std::endl;
        std::cout << "Precomputing f(k) and g(k)..." << std::endl;
    }
    chg_params->compute_f_g(J);
    for (int it = 0; it < sample_iters; ++it) {
        if (verbose)
            std::cout << "Iteration " << chg_hist.size()+1 << "..."
                << std::endl;
        double tot_time = 0;
        double row_time = 0;
        double col_time = 0;
        double swap_time = 0;
        tot_time -= clock();
        if (!col_only) {
            row_time -= clock();
            if (row_sample_time_block > 0 && it % 2 == 0)
                sample_rows<Model>(X,Z,T,J,*params,*chg_params,
                        row_sample_time_block,row_sample_time_block);
            else if (row_sample_time_block > 0 && it % 2 == 1)
                sample_rows<Model>(X,Z,T,J,*params,*chg_params,
                        (row_sample_time_block+1)/2,row_sample_time_block);
            else
                sample_rows<Model>(X,Z,T,J,*params,*chg_params);
            row_time += clock();
        }
        if (!row_only) {
            col_time -= clock();
            sample_columns<Model>(X,Z,T,J,*params,*chg_params,
                    col_sample_approx);
            col_time += clock();
        }
        if (nswaps > 0) {
            swap_time -= clock();
            metropolis_swap_change_times<Model>(X,Z,T,J,*params,nswaps);
            swap_time += clock();
        }
        chg_hist.push_back(Changes());
        for (int t = 1; t < T; ++t) {
            Changes::iterator iter = chg_hist.back().insert(
                    std::make_pair(t,std::vector<int>())).first;
            for (int j = 0; j < J; ++j) {
                if (Z[j*T+t])
                    iter->second.push_back(j);
            }
            if (iter->second.size() == 0)
                chg_hist.back().erase(iter);
        }
        tot_time += clock();
        if (verbose) {
            std::cout << "  Row Gibbs-sampling time:    "
                << row_time/CLOCKS_PER_SEC << std::endl;
            std::cout << "  Column Gibbs-sampling time: "
                << col_time/CLOCKS_PER_SEC << std::endl;
            std::cout << "  Metropolis-Hastings column swapping time: "
                << swap_time/CLOCKS_PER_SEC << std::endl;
            std::cout << "  Total iteration time: "
                << tot_time/CLOCKS_PER_SEC << std::endl;
        }
        if (MCEM_pos < MCEM_schedule.size()
                && it+1 == MCEM_schedule[MCEM_pos]) {
            if (verbose)
                std::cout << "MCEM Step..." << std::endl;
            unsigned MCEM_start = 0;
            if (MCEM_pos > 0)
                MCEM_start = MCEM_schedule[MCEM_pos-1];
            estimate_parameters<Model>(X,T,J,chg_hist,MCEM_start,
                    *params,*chg_params,estimate_piq);
            ++MCEM_pos;
            if (verbose) {
                std::cout << "  New model params:" << std::endl;
                if (estimate_piq) {
                    std::cout << "    pi_q:" << std::endl;
                    for (int i = 0; i < chg_params->Q; ++i)
                        std::cout << "      " << chg_params->q_vals[i]
                            << ": " << exp(chg_params->log_piq[i])
                            << std::endl;
                }
                for (int k = 0; k < params->K; ++k)
                    std::cout << "    " << params->names[k] << ": "
                        << params->vals[k] << std::endl;
                if (estimate_piq)
                    std::cout << "Recomputing f(k) and g(k)..."
                        << std::endl;
            }
            if (estimate_piq)
                chg_params->compute_f_g(J);
        }
    }
}

/****************** POSTERIOR MODE PROCEDURES ********************/
template <typename Model>
bool maximize_rows(const double* X,
        bool* Z, int T, int J, const ModelParams& params,
        const ChangeParams& chg_params, int first_block=-1, int block=-1) {
    /* Count number of changepoints at each time */
    int* N = new int[T];
    for (int t = 1; t < T; ++t) {
        N[t] = 0;
        for (int j = 0; j < J; ++j)
            N[t] += (int)Z[j*T+t];
    }
    double* logc = new double[T];
    double* vals = new double[T];
    int* prev = new int[T];
    Model prev_model(params);
    Model next_model(params);
    bool changed = false;
    for (int j = 0; j < J; ++j) {
        prev_model.clear();
        next_model.clear();
        for (int t = 1; t < T; ++t) {
            N[t] -= (int)Z[j*T+t];
            logc[t] = chg_params.log_f[N[t]+1]-chg_params.log_g[N[t]+1];
        }
        int start = 1;
        int end = (first_block > 0 && first_block < T ? first_block : T);
        int next = end;
        for (int t = end; t < T; ++t) {
            next = t;
            if (Z[j*T+t]) break;
            next_model.add_data(X[j*T+t]);
        }
        while (start < T) {
            typename std::list<std::pair<Model, int> > models;
            models.push_back(std::make_pair(Model(params), start - 1));
            models.back().first.combine(prev_model);
            models.back().first.add_data(X[j*T+start-1]);
            vals[start - 1] = models.back().first.logmarginal();
            for (int t = start; t < end; ++t) {
                models.push_back(std::make_pair(Model(params), t));
                double max_val = NEGINF;
                int max_ind = -1;
                for (typename std::list<std::pair<Model, int> >::iterator
                        iter = models.begin(); iter != models.end();
                        ++iter) {
                    iter->first.add_data(X[j*T+t]);
                    if (t == end - 1)
                        iter->first.combine(next_model);
                    double val = iter->first.logmarginal();
                    if (iter->second > start - 1)
                        val += vals[iter->second - 1] + logc[iter->second]
                            - logz(1-exp(logc[iter->second]));
                    if (val > max_val) {
                        max_val = val;
                        max_ind = iter->second;
                    }
                }
                vals[t] = max_val;
                prev[t] = max_ind;
            }
            int ind = prev[end - 1];
            std::list<int> changes;
            while (ind > start - 1) {
                changes.push_front(ind);
                if (ind > start)
                    ind = prev[ind - 1];
                else
                    ind = -1;
            }
            std::list<int>::iterator changes_iter = changes.begin();
            for (int t = start; t < end; ++t) {
                prev_model.add_data(X[j*T+t-1]);
                if (changes_iter != changes.end() && t == *changes_iter) {
                    if (!Z[j*T+t])
                        changed = true;
                    Z[j*T+t] = true;
                    ++N[t];
                    prev_model.clear();
                    ++changes_iter;
                } else {
                    if (Z[j*T+t])
                        changed = true;
                    Z[j*T+t] = false;
                }
            }
            start = end;
            if (block > 0 && end + block > next) {
                end = (end + block < T ? end + block : T);
                next_model.clear();
                for (int t = end; t < T; ++t) {
                    next = t;
                    if (Z[j*T+t]) break;
                    next_model.add_data(X[j*T+t]);
                }
            } else if (block > 0) {
                for (int t = end; t < end + block; ++t)
                    next_model.remove_data(X[j*T+t]);
                end += block;
            }
        }
    }
    delete[] N;
    delete[] logc;
    delete[] vals;
    delete[] prev;
    return changed;
}

template <typename Model>
bool maximize_columns(const double* X,
        bool* Z, int T, int J, const ModelParams& params,
        const ChangeParams& chg_params) {
    Model** model_before = new Model*[J];
    Model** model_after = new Model*[J];
    Model** model_all = new Model*[J];
    for (int j = 0; j < J; ++j) {
        model_before[j] = new Model(params);
        model_after[j] = new Model(params);
        model_all[j] = new Model(params);
        model_before[j]->add_data(X[j*T]);
        model_all[j]->add_data(X[j*T]);
        for (int t = 1; t < T; ++t) {
            if (Z[j*T+t])
                break;
            model_all[j]->add_data(X[j*T+t]);
            model_after[j]->add_data(X[j*T+t]);
        }
    }
    std::vector<std::pair<double, int> > diffs(J);
    bool changed = false;
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < J; ++j) {
            if (Z[j*T+t]) {
                model_after[j]->add_data(X[j*T+t]);
                model_all[j]->add_data(X[j*T+t]);
                for (int s = t+1; s < T; ++s) {
                    if (Z[j*T+s])
                        break;
                    model_after[j]->add_data(X[j*T+s]);
                    model_all[j]->add_data(X[j*T+s]);
                }
            }
            diffs[j].first = model_before[j]->logmarginal()
                + model_after[j]->logmarginal()
                - model_all[j]->logmarginal();
            diffs[j].second = j;
        }
        std::sort(diffs.begin(), diffs.end());
        double max_val = 0;
        int max_nchg = 0;
        double val = 0;
        for (int nchg = 1; nchg <= J; ++nchg) {
            val += diffs[J-nchg].first + chg_params.log_f[nchg]
                - chg_params.log_f[nchg-1];
            if (val > max_val) {
                max_val = val;
                max_nchg = nchg;
            }
        }
        for (int nchg = 1; nchg <= J; ++nchg) {
            int j = diffs[J-nchg].second;
            if (nchg <= max_nchg) {
                if (!Z[j*T+t])
                    changed = true;
                Z[j*T+t] = true;
                model_before[j]->clear();
                model_all[j]->clear();
                model_all[j]->combine(*model_after[j]);
            } else {
                if (Z[j*T+t])
                    changed = true;
                Z[j*T+t] = false;
            }
            model_before[j]->add_data(X[j*T+t]);
            model_after[j]->remove_data(X[j*T+t]);
        }
    }
    for (int j = 0; j < J; ++j) {
        delete[] model_before[j];
        delete[] model_after[j];
        delete[] model_all[j];
    }
    delete[] model_before;
    delete[] model_after;
    delete[] model_all;
    return changed;
}

template <typename Model>
bool greedy_swap_change_times(
        const double* X, bool* Z, int T, int J, const ModelParams& params) {
    Model** models = new Model*[T*J];
    Model** model_ptrs = new Model*[T*J];
    int* chg_times = new int[T];
    int** chg_ptrs = new int*[T];
    for (int t = 0; t < T; ++t)
        chg_ptrs[t] = NULL;
    int model_idx = 0;
    models[0] = new Model(params);
    int chg_idx = 0;
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; ++t) {
            if (Z[j*T+t]) {
                if (t == 0) {
                    throw std::runtime_error(
                            "Cannot have change at time 0");
                }
                ++model_idx;
                models[model_idx] = new Model(params);
                if (!chg_ptrs[t]) {
                    chg_times[chg_idx] = t;
                    chg_ptrs[t] = &(chg_times[chg_idx]);
                    ++chg_idx;
                }
            }
            models[model_idx]->add_data(X[j*T+t]);
            model_ptrs[j*T+t] = models[model_idx];
        }
        ++model_idx;
        models[model_idx] = new Model(params);
    }
    if (chg_idx == 0 || T < 3) {
        for (int i = 0; i <= model_idx; ++i)
            delete models[i];
        delete[] models;
        delete[] model_ptrs;
        delete[] chg_times;
        delete[] chg_ptrs;
        return false;
    }
    bool changed = false;
    for (int i = 0; i < chg_idx; ++i) {
        int t = chg_times[i];
        double pminus = 0;
        double pplus = 0;
        if (t > 1)
            pminus = evaluate_swap<Model>(t,t-1,X,Z,T,J,model_ptrs);
        if (t < T-1)
            pplus = evaluate_swap<Model>(t,t+1,X,Z,T,J,model_ptrs);
        if (pminus > 1 && pminus > pplus) {
            changed = true;
            while (pminus > 1) {
                perform_swap<Model>(t,t-1,X,Z,T,J,model_ptrs,chg_ptrs);
                t = t-1;
                pminus = 0;
                if (t > 1)
                    pminus = evaluate_swap<Model>(t,t-1,X,Z,T,J,
                            model_ptrs);
            }
        } else if (pplus > 1 && pplus > pminus) {
            changed = true;
            while (pplus > 1) {
                perform_swap<Model>(t,t+1,X,Z,T,J,model_ptrs,chg_ptrs);
                t = t+1;
                pplus = 0;
                if (t < T-1)
                    pplus = evaluate_swap<Model>(t,t+1,X,Z,T,J,model_ptrs);
            }
        }
    }
    for (int i = 0; i <= model_idx; ++i)
        delete models[i];
    delete[] models;
    delete[] model_ptrs;
    delete[] chg_times;
    delete[] chg_ptrs;
    return changed;
}

template <typename Model>
void bayesian_simultaneous_changepoint::compute_posterior_mode(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, int max_iters, int row_time_block, 
        bool col_only, bool row_only, bool swap, bool verbose) {
    if (!params) {
        params = Model::init_params(X, T, J);
    }
    if (!chg_params) {
        chg_params = init_change_params(J);
    }
    check_model_params<Model>(params);
    check_change_params(chg_params);
    if (verbose) {
        std::cout << "Initial model params:" << std::endl;
        std::cout << "  pi_q:" << std::endl;
        for (int i = 0; i < chg_params->Q; ++i)
            std::cout << "    " << chg_params->q_vals[i]
                << ": " << exp(chg_params->log_piq[i])
                << std::endl;
        for (int k = 0; k < params->K; ++k)
            std::cout << "  " << params->names[k] << ": "
                << params->vals[k] << std::endl;
        std::cout << "Precomputing f(k) and g(k)..." << std::endl;
    }
    chg_params->compute_f_g(J);
    for (int iter = 1; iter <= max_iters; ++iter) {
        bool changed = false;
        if (verbose)
            std::cout << "Iteration " << iter << std::endl;
        double tot_time = 0;
        double row_time = 0;
        double col_time = 0;
        double swap_time = 0;
        tot_time -= clock();
        if (!col_only) {
            row_time -= clock();
            if (row_time_block > 0 && iter % 2 == 0)
                changed |= maximize_rows<Model>(X,Z,T,J,*params,
                        *chg_params,row_time_block,row_time_block);
            else if (row_time_block > 0 && iter % 2 == 1)
                changed |= maximize_rows<Model>(X,Z,T,J,*params,
                        *chg_params,(row_time_block+1)/2,row_time_block);
            else
                changed |= maximize_rows<Model>(X,Z,T,J,*params,
                        *chg_params);
            row_time += clock();
        }
        if (!row_only) {
            col_time -= clock();
            changed |= maximize_columns<Model>(X,Z,T,J,*params,
                    *chg_params);
            col_time += clock();
        }
        if (swap) {
            swap_time -= clock();
            changed |= greedy_swap_change_times<Model>(X,Z,T,J,*params);
            swap_time += clock();
        }
        tot_time += clock();
        if (verbose) {
            std::cout << "  Row maximization time:    "
                << row_time/CLOCKS_PER_SEC << std::endl;
            std::cout << "  Column maximization time: "
                << col_time/CLOCKS_PER_SEC << std::endl;
            std::cout << "  Column adjustment time: "
                << swap_time/CLOCKS_PER_SEC << std::endl;
            std::cout << "  Total iteration time: "
                << tot_time/CLOCKS_PER_SEC << std::endl;
        }
        if (!changed) {
            if (verbose) {
                std::cout << "Converged" << std::endl;
            }
            return;
        }
    }
    std::cerr << "WARNING: posterior maximization did not converge"
        << std::endl;
}


/********** Explicit template specializations, for the linker **********/
template
void bayesian_simultaneous_changepoint::gibbs_sample<BernoulliModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, ChangeHist& chg_hist,
        int sample_iters, const std::vector<int>& MCEM_schedule,
        bool estimate_piq, int row_sample_time_block,
        double col_sample_approx, bool col_only, bool row_only, int nswaps,
        bool verbose);
template
void bayesian_simultaneous_changepoint::gibbs_sample<LaplaceScaleModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, ChangeHist& chg_hist,
        int sample_iters, const std::vector<int>& MCEM_schedule,
        bool estimate_piq, int row_sample_time_block,
        double col_sample_approx, bool col_only, bool row_only, int nswaps,
        bool verbose);
template
void bayesian_simultaneous_changepoint::gibbs_sample<NormalMeanModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, ChangeHist& chg_hist,
        int sample_iters, const std::vector<int>& MCEM_schedule,
        bool estimate_piq, int row_sample_time_block,
        double col_sample_approx, bool col_only, bool row_only, int nswaps,
        bool verbose);
template
void bayesian_simultaneous_changepoint::gibbs_sample<NormalMeanVarModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, ChangeHist& chg_hist,
        int sample_iters, const std::vector<int>& MCEM_schedule,
        bool estimate_piq, int row_sample_time_block,
        double col_sample_approx, bool col_only, bool row_only, int nswaps,
        bool verbose);
template
void bayesian_simultaneous_changepoint::gibbs_sample<NormalVarModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, ChangeHist& chg_hist,
        int sample_iters, const std::vector<int>& MCEM_schedule,
        bool estimate_piq, int row_sample_time_block,
        double col_sample_approx, bool col_only, bool row_only, int nswaps,
        bool verbose);
template
void bayesian_simultaneous_changepoint::gibbs_sample<PoissonModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, ChangeHist& chg_hist,
        int sample_iters, const std::vector<int>& MCEM_schedule,
        bool estimate_piq, int row_sample_time_block,
        double col_sample_approx, bool col_only, bool row_only, int nswaps,
        bool verbose);
template
void bayesian_simultaneous_changepoint::compute_posterior_mode<BernoulliModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, int max_iters, int row_time_block, 
        bool col_only, bool row_only, bool swap, bool verbose);
template
void bayesian_simultaneous_changepoint::compute_posterior_mode<LaplaceScaleModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, int max_iters, int row_time_block, 
        bool col_only, bool row_only, bool swap, bool verbose);
template
void bayesian_simultaneous_changepoint::compute_posterior_mode<NormalMeanModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, int max_iters, int row_time_block, 
        bool col_only, bool row_only, bool swap, bool verbose);
template
void bayesian_simultaneous_changepoint::compute_posterior_mode<NormalMeanVarModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, int max_iters, int row_time_block, 
        bool col_only, bool row_only, bool swap, bool verbose);
template
void bayesian_simultaneous_changepoint::compute_posterior_mode<NormalVarModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, int max_iters, int row_time_block, 
        bool col_only, bool row_only, bool swap, bool verbose);
template
void bayesian_simultaneous_changepoint::compute_posterior_mode<PoissonModel>(
        const double* X, bool* Z, int T, int J, ModelParams*& params,
        ChangeParams*& chg_params, int max_iters, int row_time_block, 
        bool col_only, bool row_only, bool swap, bool verbose);
