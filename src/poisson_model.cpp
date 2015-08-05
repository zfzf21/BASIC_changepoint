#include "poisson_model.hpp"
#include <sstream>

using namespace bayesian_simultaneous_changepoint;

PoissonModel::PoissonModel(const ModelParams& params) {
    n = 0;
    xsum = 0;
    logxfactsum = 0;
    reset_params(params);
}

void PoissonModel::reset_params(const ModelParams& params) {
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

void PoissonModel::add_data(double x) {
    n += 1;
    xsum += x;
    logxfactsum += lgamma(x+1);
}

void PoissonModel::remove_data(double x) {
    n -= 1;
    xsum -= x;
    logxfactsum -= lgamma(x+1);
}

void PoissonModel::clear() {
    n = 0;
    xsum = 0;
    logxfactsum = 0;
}

void PoissonModel::combine(const PoissonModel& other) {
    n += other.n;
    xsum += other.xsum;
    logxfactsum += other.logxfactsum;
}

double PoissonModel::logmarginal() {
    return -logxfactsum+a*log(b)-lgamma(a)
        +lgamma(a+xsum)-(a+xsum)*log(b+n);
}

double PoissonModel::lower_bound(const std::string& param_name) {
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

double PoissonModel::upper_bound(const std::string& param_name) {
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

ModelParams* PoissonModel::init_params(const double* X, int T, int J) {
    int nsegs = (T-1)/100+1;
    double* means = new double[J*nsegs];
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; t += 100) {
            double xsum = 0;
            int n = (t + 100 < T ? 100 : T-t);
            for (int s = t; s < t+n; ++s) {
                xsum += X[j*T+s];
            }
            means[j*nsegs+t/100] = xsum/n;
        }
    }
    double means_sum = 0;
    double means_sqsum = 0;
    for (int i = 0; i < J*nsegs; ++i) {
        means_sum += means[i];
        means_sqsum += means[i]*means[i];
    }
    ModelParams* params = new ModelParams();
    params->K = 2;
    params->names = new std::string[2];
    params->names[0] = "alpha";
    params->names[1] = "beta";
    params->vals = new double[2];
    double means_var = means_sqsum/(J*nsegs)
        -(means_sum/(J*nsegs))*(means_sum/(J*nsegs));
    params->vals[0] = (means_sum/(J*nsegs))*(means_sum/(J*nsegs)) /
        (means_var > 0 ? means_var : 1);
    params->vals[1] = params->vals[0] / (means_sum > 0 ?
            means_sum/(J*nsegs) : 1);
    delete[] means;
    return params;
}
