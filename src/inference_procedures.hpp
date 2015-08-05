#ifndef __bayesian_simultaneous_changepoint_inference_procedures_hpp__
#define __bayesian_simultaneous_changepoint_inference_procedures_hpp__

#include "base.hpp"
#include <vector>
#include <map>

namespace bayesian_simultaneous_changepoint {

    typedef std::map<int, std::vector<int> > Changes;
    typedef std::vector<Changes> ChangeHist;

    void set_seed(int s);

    /* Main Gibbs sampling + MCEM update routine */
    template <typename Model>
        void gibbs_sample(const double* X, bool* Z, int T, int J,
                ModelParams*& params, ChangeParams*& chg_params,
                ChangeHist& chg_hist, int sample_iters=100,
                const std::vector<int>& MCEM_schedule=std::vector<int>(),
                bool estimate_piq=true, int row_sample_time_block=50,
                double col_sample_approx=1e-6,
                bool col_only=false, bool row_only=false,
                int nswaps=-1, bool verbose=true);

    /* Main posterior-maximization routine */
    template <typename Model>
        void compute_posterior_mode(const double* X, bool* Z, int T, int J,
                ModelParams*& params, ChangeParams*& chg_params,
                int max_iters=100, int row_time_block=50,
                bool col_only=false, bool row_only=false, bool swap=true,
                bool verbose=true);

    /* Helper functions -- expose here only for debugging
    template <typename Model>
        void sample_rows(const double* X, bool* Z, int T, int J,
                const ModelParams& params, const ChangeParams& chg_params, 
                int first_block=-1, int block=-1);
    template <typename Model>
        void sample_columns(const double* X, bool* Z, int T, int J,
                const ModelParams& params, const ChangeParams& chg_params, 
                double approx=-1);
    template <typename Model>
        void metropolis_swap_change_times(const double* X, bool* Z, int T,
                int J, const ModelParams& params, int nswaps);
    template <typename Model>
        void estimate_parameters(const double* X, int T, int J,
                const ChangeHist& chg_hist, unsigned MCEM_start,
                ModelParams& params, ChangeParams& chg_params,
                bool update_piq=true);
    template <typename Model>
        bool maximize_rows(const double* X, bool* Z, int T, int J,
                const ModelParams& params, const ChangeParams& chg_params,
                int first_block=-1, int block=-1);
    template <typename Model>
        bool maximize_columns(const double* X, bool* Z, int T, int J,
                const ModelParams& params, const ChangeParams& chg_params);
    template <typename Model>
        bool greedy_swap_change_times(const double* X, bool* Z, int T,
                int J, const ModelParams& params);
                */
}

#endif
