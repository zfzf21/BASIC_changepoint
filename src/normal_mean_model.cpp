#include "normal_mean_model.hpp"
#include <sstream>

using namespace bayesian_simultaneous_changepoint;

NormalMeanModel::NormalMeanModel(const ModelParams& params) {
    n = 0;
    xsum = 0;
    xsumsq = 0;
    reset_params(params);
}

void NormalMeanModel::reset_params(const ModelParams& params) {
    bool found_var = false;
    bool found_mu0 = false;
    bool found_lamb = false;
    for (int k = 0; k < params.K; ++k) {
        if (params.names[k] == "mu_0") {
            mu0 = params.vals[k];
            found_mu0 = true;
        } else if (params.names[k] == "sigma^2") {
            var = params.vals[k];
            found_var = true;
        } else if (params.names[k] == "lambda") {
            lamb = params.vals[k];
            found_lamb = true;
        } else {
            std::stringstream ss;
            ss << "Unknown parameter " << params.names[k];
            throw std::runtime_error(ss.str());
        }
    }
    if (!found_mu0)
        throw std::runtime_error("Parameter mu_0 not found");
    if (!found_var)
        throw std::runtime_error("Parameter sigma^2 not found");
    if (!found_lamb)
        throw std::runtime_error("Parameter lambda not found");
}

void NormalMeanModel::add_data(double x) {
    n += 1;
    xsum += x;
    xsumsq += x*x;
}

void NormalMeanModel::remove_data(double x) {
    n -= 1;
    xsum -= x;
    xsumsq -= x*x;
}

void NormalMeanModel::clear() {
    n = 0;
    xsum = 0;
    xsumsq = 0;
}

void NormalMeanModel::combine(const NormalMeanModel& other) {
    n += other.n;
    xsum += other.xsum;
    xsumsq += other.xsumsq;
}

double NormalMeanModel::logmarginal() {
    return -n/2*log(2*PI*var)+0.5*(log(lamb)-log(lamb+n))
        -(lamb*mu0*mu0+xsumsq-(lamb*mu0+xsum)
                *(lamb*mu0+xsum)/(lamb+n))/(2*var);
}

double NormalMeanModel::lower_bound(const std::string& param_name) {
    if (param_name == "mu_0")
        return -1e10;
    else if (param_name == "sigma^2")
        return 1e-10;
    else if (param_name == "lambda")
        return 1e-10;
    else {
        std::stringstream ss;
        ss << "Unknown parameter " << param_name;
        throw std::runtime_error(ss.str());
    }
}

double NormalMeanModel::upper_bound(const std::string& param_name) {
    if (param_name == "mu_0")
        return 1e10;
    else if (param_name == "sigma^2")
        return 1e10;
    else if (param_name == "lambda")
        return 1e10;
    else {
        std::stringstream ss;
        ss << "Unknown parameter " << param_name;
        throw std::runtime_error(ss.str());
    }
}

ModelParams* NormalMeanModel::init_params(const double* X, int T, int J) {
    int nsegs = (T-1)/100+1;
    double* means = new double[J*nsegs];
    double* vars = new double[J*nsegs];
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; t += 100) {
            double xsum = 0;
            double xsqsum = 0;
            int n = (t + 100 < T ? 100 : T-t);
            for (int s = t; s < t+n; ++s) {
                xsum += X[j*T+s];
                xsqsum += X[j*T+s]*X[j*T+s];
            }
            means[j*nsegs+t/100] = xsum/n;
            vars[j*nsegs+t/100] = xsqsum/n-xsum*xsum/(n*n);
        }
    }
    double means_sum = 0;
    double means_sqsum = 0;
    double vars_sum = 0;
    for (int i = 0; i < J*nsegs; ++i) {
        means_sum += means[i];
        means_sqsum += means[i]*means[i];
        vars_sum += vars[i];
    }
    ModelParams* params = new ModelParams();
    params->K = 3;
    params->names = new std::string[3];
    params->names[0] = "mu_0";
    params->names[1] = "sigma^2";
    params->names[2] = "lambda";
    params->vals = new double[3];
    params->vals[0] = means_sum/(J*nsegs);
    params->vals[1] = vars_sum/(J*nsegs);
    double means_var = means_sqsum/(J*nsegs)
        - (means_sum/(J*nsegs)) * (means_sum/(J*nsegs));
    params->vals[2] = params->vals[1] / (means_var > 0 ? means_var : 1);
    delete[] means;
    delete[] vars;
    return params;
}
