#ifndef __bayesian_simultaneous_changepoint_base_hpp__
#define __bayesian_simultaneous_changepoint_base_hpp__

#include <math.h>
#include <limits>
#include <string>

namespace bayesian_simultaneous_changepoint {

    const double PI = 3.14159265359;
    const double NEGINF = -std::numeric_limits<double>::max();

    /* log(e^a+e^b) */
    inline double logaddexp(double a, double b) {
        return (a>b ? a : b) + log(1 + exp(-fabs(b-a)));
    }
    /* log(q), returning -inf if q is 0 */
    inline double logz(double q) {
        return (q<1e-100 ? NEGINF : log(q));
    }

    /* Changepoint distribution parameters */
    struct ChangeParams {
        int Q; // Number of support points
        double* q_vals; // Support points
        double* log_piq; // Log-probabilities of support points
        double* log_f;
        double* log_g;
        ChangeParams();
        void compute_f_g(int J); // Precompute f and g for inference
        void clear_f_g();
    };
    /* Data distribution model and prior parameter definitions */
    struct ModelParams {
        int K; // Number of parameters
        std::string* names; // Parameter names
        double* vals; // Parameter values
    };
}

#endif
