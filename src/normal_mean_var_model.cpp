#include "normal_mean_var_model.hpp"
#include <sstream>

using namespace bayesian_simultaneous_changepoint;

NormalMeanVarModel::NormalMeanVarModel(const ModelParams& params) {
    n = 0;
    xsum = 0;
    xsumsq = 0;
    reset_params(params);
}

void NormalMeanVarModel::reset_params(const ModelParams& params) {
    bool found_mu0 = false;
    bool found_lamb = false;
    bool found_a = false;
    bool found_b = false;
    for (int k = 0; k < params.K; ++k) {
        if (params.names[k] == "mu_0") {
            mu0 = params.vals[k];
            found_mu0 = true;
        } else if (params.names[k] == "lambda") {
            lamb = params.vals[k];
            found_lamb = true;
        } else if (params.names[k] == "alpha") {
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
    if (!found_mu0)
        throw std::runtime_error("Parameter mu_0 not found");
    if (!found_lamb)
        throw std::runtime_error("Parameter lambda not found");
    if (!found_a)
        throw std::runtime_error("Parameter alpha not found");
    if (!found_b)
        throw std::runtime_error("Parameter beta not found");
}

void NormalMeanVarModel::add_data(double x) {
    n += 1;
    xsum += x;
    xsumsq += x*x;
}

void NormalMeanVarModel::remove_data(double x) {
    n -= 1;
    xsum -= x;
    xsumsq -= x*x;
}

void NormalMeanVarModel::clear() {
    n = 0;
    xsum = 0;
    xsumsq = 0;
}

void NormalMeanVarModel::combine(const NormalMeanVarModel& other) {
    n += other.n;
    xsum += other.xsum;
    xsumsq += other.xsumsq;
}

double NormalMeanVarModel::logmarginal() {
    return -n/2*log(2*PI)+0.5*(log(lamb)-log(lamb+n))
        +a*log(b)-lgamma(a)+lgamma(a+n/2)
        -(a+n/2)*log(b+(lamb*mu0*mu0+xsumsq)/2-(lamb*mu0
                    +xsum)*(lamb*mu0+xsum)/(2*(lamb+n)));
}

double NormalMeanVarModel::lower_bound(const std::string& param_name) {
    if (param_name == "mu_0")
        return -1e10;
    else if (param_name == "lambda")
        return 1e-10;
    else if (param_name == "alpha")
        return 1e-10;
    else if (param_name == "beta")
        return 1e-10;
    else {
        std::stringstream ss;
        ss << "Unknown parameter " << param_name;
        throw std::runtime_error(ss.str());
    }
}

double NormalMeanVarModel::upper_bound(const std::string& param_name) {
    if (param_name == "mu_0")
        return 1e10;
    else if (param_name == "lambda")
        return 1e10;
    else if (param_name == "alpha")
        return 1e10;
    else if (param_name == "beta")
        return 1e10;
    else {
        std::stringstream ss;
        ss << "Unknown parameter " << param_name;
        throw std::runtime_error(ss.str());
    }
}

ModelParams* NormalMeanVarModel::init_params(const double* X,int T,int J) {
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
    double vars_sqsum = 0;
    for (int i = 0; i < J*nsegs; ++i) {
        means_sum += means[i];
        means_sqsum += means[i]*means[i];
        vars_sum += vars[i];
        vars_sqsum += vars[i]*vars[i];
    }
    ModelParams* params = new ModelParams();
    params->K = 4;
    params->names = new std::string[4];
    params->names[0] = "mu_0";
    params->names[1] = "lambda";
    params->names[2] = "alpha";
    params->names[3] = "beta";
    params->vals = new double[4];
    params->vals[0] = means_sum/(J*nsegs);
    params->vals[1] = vars_sum/(J*nsegs) /
        (means_sqsum/(J*nsegs)-means_sum*means_sum/(J*J*nsegs*nsegs));
    params->vals[2] = vars_sum*vars_sum/(J*J*nsegs*nsegs) / 
        (vars_sqsum/(J*nsegs)-vars_sum*vars_sum/(J*J*nsegs*nsegs)) + 2;
    params->vals[3] = vars_sum/(J*nsegs)*(params->vals[2]-1);
    delete[] means;
    delete[] vars;
    return params;
}
