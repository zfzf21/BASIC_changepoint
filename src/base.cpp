#include "base.hpp"

using namespace bayesian_simultaneous_changepoint;

ChangeParams::ChangeParams() {
    log_f = NULL;
    log_g = NULL;
}

void ChangeParams::compute_f_g(int J) {
    clear_f_g();
    log_f = new double[J+1];
    log_g = new double[J+1];
    for (int j = 0; j < J+1; ++j) {
        log_f[j] = NEGINF;
        log_g[j] = NEGINF;
    }
    for (int i = 0; i < Q; ++i) {
        double logq = logz(q_vals[i]);
        double logp = logz(1.0-q_vals[i]);
        double log_factor = J*logp+log_piq[i];
        log_f[0] = logaddexp(log_f[0], log_factor);
        for (int j = 1; j <= J; ++j) {
            log_factor -= logp;
            log_g[j] = logaddexp(log_g[j], log_factor);
            log_factor += logq;
            log_f[j] = logaddexp(log_f[j], log_factor);
        }
    }
}

void ChangeParams::clear_f_g() {
    if (log_f) {
        delete[] log_f;
        log_f = NULL;
    }
    if (log_g) {
        delete[] log_g;
        log_g = NULL;
    }
}
