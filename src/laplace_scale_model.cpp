#include "laplace_scale_model.hpp"
#include <sstream>

using namespace bayesian_simultaneous_changepoint;

LaplaceScaleModel::LaplaceScaleModel(const ModelParams& params) {
    n = 0;
    xabssum = 0;
    reset_params(params);
}

void LaplaceScaleModel::reset_params(const ModelParams& params) {
    bool found_a = false;
    bool found_b = false;
    for (int k = 0; k < params.K; ++k) {
        if (params.names[k] == "alpha") {
            a = params.vals[k];
            found_a = true;
        } else if (params.names[k] == "beta") {
            b = params.vals[k];
            found_b = true;
        } else {
            std::stringstream ss;
            ss << "Unknown parameter " << params.names[k];
            throw std::runtime_error(ss.str());
        }
    }
    if (!found_a)
        throw std::runtime_error("Parameter alpha not found");
    if (!found_b)
        throw std::runtime_error("Parameter beta not found");
}

void LaplaceScaleModel::add_data(double x) {
    n += 1;
    xabssum += fabs(x);
}

void LaplaceScaleModel::remove_data(double x) {
    n -= 1;
    xabssum -= fabs(x);
}

void LaplaceScaleModel::clear() {
    n = 0;
    xabssum = 0;
}

void LaplaceScaleModel::combine(const LaplaceScaleModel& other) {
    n += other.n;
    xabssum += other.xabssum;
}

double LaplaceScaleModel::logmarginal() {
    return -n*log(2.0)+a*log(b)-lgamma(a)+lgamma(a+n)
        -(a+n)*log(b+xabssum);
}

double LaplaceScaleModel::lower_bound(const std::string& param_name) {
    if (param_name == "alpha")
        return 1e-10;
    else if (param_name == "beta")
        return 1e-10;
    else {
        std::stringstream ss;
        ss << "Unknown parameter " << param_name;
        throw std::runtime_error(ss.str());
    }
}

double LaplaceScaleModel::upper_bound(const std::string& param_name) {
    if (param_name == "alpha")
        return 1e10;
    else if (param_name == "beta")
        return 1e10;
    else {
        std::stringstream ss;
        ss << "Unknown parameter " << param_name;
        throw std::runtime_error(ss.str());
    }
}

ModelParams* LaplaceScaleModel::init_params(const double* X, int T, int J) {
    int nsegs = (T-1)/100+1;
    double* deviations = new double[J*nsegs];
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; t += 100) {
            double xabssum = 0;
            int n = (t + 100 < T ? 100 : T-t);
            for (int s = t; s < t+n; ++s) {
                xabssum += fabs(X[j*T+s]);
            }
            deviations[j*nsegs+t/100] = xabssum/n;
        }
    }
    double deviations_sum = 0;
    double deviations_sqsum = 0;
    for (int i = 0; i < J*nsegs; ++i) {
        deviations_sum += deviations[i];
        deviations_sqsum += deviations[i]*deviations[i];
    }
    ModelParams* params = new ModelParams();
    params->K = 2;
    params->names = new std::string[2];
    params->names[0] = "alpha";
    params->names[1] = "beta";
    params->vals = new double[2];
    params->vals[0] = deviations_sum*deviations_sum/(J*J*nsegs*nsegs) /
        (deviations_sqsum/(J*nsegs)
         -deviations_sum*deviations_sum/(J*J*nsegs*nsegs)) + 2;
    params->vals[1] = deviations_sum/(J*nsegs)*(params->vals[0]-1);
    delete[] deviations;
    return params;
}
