#ifndef __bayesian_simultaneous_changepoint_laplace_scale_model_hpp__
#define __bayesian_simultaneous_changepoint_laplace_scale_model_hpp__

#include "base.hpp"

namespace bayesian_simultaneous_changepoint {
    class LaplaceScaleModel {
        private:
            double a;
            double b;
            double n;
            double xabssum;
        public:
            LaplaceScaleModel(const ModelParams& params);
            void add_data(double x);
            void remove_data(double x);
            void clear();
            void combine(const LaplaceScaleModel& other);
            double logmarginal();
            void reset_params(const ModelParams& params);

            static double lower_bound(const std::string& param_name);
            static double upper_bound(const std::string& param_name);
            static ModelParams* init_params(const double* X, int T, int J);
    };
}

#endif
