#ifndef __bayesian_simultaneous_changepoint_normal_mean_var_model_hpp__
#define __bayesian_simultaneous_changepoint_normal_mean_var_model_hpp__

#include "base.hpp"

namespace bayesian_simultaneous_changepoint {

    class NormalMeanVarModel {
        private:
            double mu0;
            double lamb;
            double a;
            double b;
            double n;
            double xsum;
            double xsumsq;

        public:
            NormalMeanVarModel(const ModelParams& params);
            void add_data(double x);
            void remove_data(double x);
            void clear();
            void combine(const NormalMeanVarModel& other);
            double logmarginal();
            void reset_params(const ModelParams& params);

            static double lower_bound(const std::string& param_name);
            static double upper_bound(const std::string& param_name);
            static ModelParams* init_params(const double* X, int T, int J);
    };

}

#endif
