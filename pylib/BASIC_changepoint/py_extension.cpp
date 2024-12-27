#include <Python.h>
#include <numpy/arrayobject.h>
#include "../../src/base.hpp"
#include "../../src/inference_procedures.hpp"
#include "../../src/bernoulli_model.hpp"
#include "../../src/laplace_scale_model.hpp"
#include "../../src/normal_mean_model.hpp"
#include "../../src/normal_mean_var_model.hpp"
#include "../../src/normal_var_model.hpp"
#include "../../src/poisson_model.hpp"
#include <sstream>
#include <cassert>

using namespace bayesian_simultaneous_changepoint;

/*********************** Utility functions ****************************/
bool py_extract_MCEM_schedule(PyObject* py_MCEM_schedule,
        std::vector<int>& MCEM_schedule) {
    MCEM_schedule.clear();
    if (!PyList_Check(py_MCEM_schedule)) {
        PyErr_SetString(PyExc_ValueError,
                "MCEM schedule must be of type 'list'");
        return false;
    }
    MCEM_schedule.resize((int)PyList_Size(py_MCEM_schedule));
    for (int i = 0; i < (int)PyList_Size(py_MCEM_schedule); ++i)
        MCEM_schedule[i] = (int)PyLong_AsLong(PyList_GetItem(
                    py_MCEM_schedule, i));
    return true;
}
void delete_params(ModelParams* p) {
    if (p) {
        delete[] p->names;
        delete[] p->vals;
        delete p;
    }
}
void delete_change_params(ChangeParams* p) {
    if (p) {
        delete[] p->q_vals;
        delete[] p->log_piq;
        p->clear_f_g();
        delete p;
    }
}
bool py_extract_params(PyObject* py_params, ModelParams*& p) {
    if (py_params == Py_None) {
        p = NULL;
        return true;
    }
    p = new ModelParams();
    p->K = int(PyDict_Size(py_params));
    p->names = new std::string[p->K];
    p->vals = new double[p->K];
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    int k = 0;
    while (PyDict_Next(py_params, &pos, &key, &value)) {
        p->names[k] = std::string(PyUnicode_AsUTF8(key));
        p->vals[k] = PyFloat_AsDouble(value);
        ++k;
    }
    if (PyErr_Occurred()) {
        delete_params(p);
        PyErr_SetString(PyExc_ValueError,"Error parsing prior parameters");
        return false;
    } else
        return true;
}
bool py_extract_change_params(PyObject* py_q_vals, PyObject* py_piq,
        ChangeParams*& p) {
    if (py_q_vals == Py_None || py_piq == Py_None) {
        p = NULL;
        return true;
    }
    p = new ChangeParams();
    p->Q = int(PyList_Size(py_q_vals));
    p->q_vals = new double[p->Q];
    p->log_piq = new double[p->Q];
    for (int i = 0; i < p->Q; ++i) {
        p->q_vals[i] = PyFloat_AsDouble(PyList_GetItem(py_q_vals, i));
        double piq = PyFloat_AsDouble(PyList_GetItem(py_piq, i));
        if (piq < 0) {
            PyErr_SetString(PyExc_ValueError,
                    "pi_q values must be nonnegative");
            return false;
        }
        p->log_piq[i] = logz(piq);
    }
    if (PyErr_Occurred()) {
        delete_change_params(p);
        PyErr_SetString(PyExc_ValueError,
                "Error parsing changepoint parameters");
        return false;
    } else
        return true;
}
bool py_check_matrices(PyObject* np_X, PyObject* np_Z, int& T, int& J) {
    if (np_X == NULL || np_Z == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix inputs");
        Py_XDECREF(np_X);
        Py_XDECREF(np_Z);
        return false;
    }
    if ((int)PyArray_NDIM(np_X) != 2 || (int)PyArray_NDIM(np_Z) != 2) {
        PyErr_SetString(PyExc_ValueError,
                "Input matrices do not have correct dimensions");
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return false;
    }
    J = (int)PyArray_DIM(np_X, 0);
    T = (int)PyArray_DIM(np_X, 1);
    if (J != PyArray_DIM(np_Z, 0) || T != PyArray_DIM(np_Z, 1)) {
        PyErr_SetString(PyExc_ValueError,
                "Input matrices are not of the same size");
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return false;
    }
    return true;
}

/*void py_extract_chg_hist(PyObject* py_chg_hist, std::vector<std::map<int,
        std::vector<int> > >& chg_hist) {
    chg_hist.clear();
    chg_hist.resize((int)PyList_Size(py_chg_hist));
    for (unsigned i = 0; i < chg_hist.size(); ++i) {
        PyObject* py_dict = PyList_GetItem(py_chg_hist, (Py_ssize_t)i);
        Py_ssize_t pos = 0;
        PyObject* key;
        PyObject* value;
        while (PyDict_Next(py_dict, &pos, &key, &value)) {
            int t = (int)PyLong_AsLong(key);
            chg_hist[i].insert(std::make_pair(t, std::vector<int>()));
            chg_hist[i][t].resize((int)PyList_Size(value));
            for (unsigned j = 0; j < chg_hist[i][t].size(); ++j)
                chg_hist[i][t][j] = (int)PyLong_AsLong(PyList_GetItem(
                            value, (Py_ssize_t)j));
        }
    }
}*/

/*************** Wrappers for sampler and maximizer *********************/
/* Seed random number generator */
static PyObject* py_seed(PyObject* self, PyObject* args) {
    int s = 0;
    if (!PyArg_ParseTuple(args, "i", &s))
        return NULL;
    set_seed(s);
    Py_INCREF(Py_None);
    return Py_None;
}

/* Gibbs sampler */
PyObject* py_gibbs_sample(PyObject* self, PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_Z = NULL;
    char* model = NULL;
    PyObject* py_params = NULL;
    PyObject* py_q_vals = NULL;
    PyObject* py_piq = NULL;
    int sample_iters = 0;
    PyObject* py_MCEM_schedule = NULL;
    PyObject* py_estimate_piq = NULL;
    int row_sample_time_block = 0;
    double col_sample_approx = 0;
    PyObject* py_col_only = NULL;
    PyObject* py_row_only = NULL;
    int nswaps = 0;
    PyObject* py_verbose = NULL;
    if (!PyArg_ParseTuple(args, "OOsOOOiOOidOOiO", &py_X,&py_Z,&model,
                &py_params,&py_q_vals,&py_piq,&sample_iters,
                &py_MCEM_schedule,&py_estimate_piq,
                &row_sample_time_block,&col_sample_approx,&py_col_only,
                &py_row_only,&nswaps,&py_verbose))
        return NULL;
    bool estimate_piq = (bool)PyObject_IsTrue(py_estimate_piq);
    bool col_only = (bool)PyObject_IsTrue(py_col_only);
    bool row_only = (bool)PyObject_IsTrue(py_row_only);
    bool verbose = (bool)PyObject_IsTrue(py_verbose);
    std::vector<int> MCEM_schedule;
    if (!py_extract_MCEM_schedule(py_MCEM_schedule, MCEM_schedule))
        return NULL;
    ChangeParams* chg_params = NULL;
    if (!py_extract_change_params(py_q_vals, py_piq, chg_params))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params)) {
        delete_change_params(chg_params);
        return NULL;
    }
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject* np_Z = PyArray_FROM_OTF(py_Z, NPY_BOOL, NPY_INOUT_ARRAY);
    int T;
    int J;
    if (!py_check_matrices(np_X, np_Z, T, J)) {
        delete_change_params(chg_params);
        delete_params(params);
        return NULL;
    }
    double* X = (double*) PyArray_DATA(np_X);
    bool* Z = (bool*) PyArray_DATA(np_Z);
    ChangeHist chg_hist;
    try {
        if (strcmp(model, "normal_mean") == 0) {
            gibbs_sample<NormalMeanModel>(X,Z,T,J,params,chg_params,
                    chg_hist,sample_iters,MCEM_schedule,
                    estimate_piq,row_sample_time_block,
                    col_sample_approx,col_only,row_only,nswaps,verbose);
        } else if (strcmp(model, "normal_mean_var") == 0) {
            gibbs_sample<NormalMeanVarModel>(X,Z,T,J,params,chg_params,
                    chg_hist,sample_iters,MCEM_schedule,
                    estimate_piq,row_sample_time_block,
                    col_sample_approx,col_only,row_only,nswaps,verbose);
        } else if (strcmp(model, "normal_var") == 0) {
            gibbs_sample<NormalVarModel>(X,Z,T,J,params,chg_params,
                    chg_hist,sample_iters,MCEM_schedule,
                    estimate_piq,row_sample_time_block,
                    col_sample_approx,col_only,row_only,nswaps,verbose);
        } else if (strcmp(model, "poisson") == 0) {
            gibbs_sample<PoissonModel>(X,Z,T,J,params,chg_params,
                    chg_hist,sample_iters,MCEM_schedule,
                    estimate_piq,row_sample_time_block,
                    col_sample_approx,col_only,row_only,nswaps,verbose);
        } else if (strcmp(model, "bernoulli") == 0) {
            gibbs_sample<BernoulliModel>(X,Z,T,J,params,chg_params,
                    chg_hist,sample_iters,MCEM_schedule,
                    estimate_piq,row_sample_time_block,
                    col_sample_approx,col_only,row_only,nswaps,verbose);
        } else if (strcmp(model, "laplace_scale") == 0) {
            gibbs_sample<LaplaceScaleModel>(X,Z,T,J,params,chg_params,
                    chg_hist,sample_iters,MCEM_schedule,
                    estimate_piq,row_sample_time_block,
                    col_sample_approx,col_only,row_only,nswaps,verbose);
        } else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_change_params(chg_params);
            delete_params(params);
            Py_DECREF(np_X);
            Py_DECREF(np_Z);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_change_params(chg_params);
        delete_params(params);
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return NULL;
    }
    PyObject* py_new_params = PyDict_New();
    for (int k = 0; k < params->K; ++k) {
        PyObject* py_val = PyFloat_FromDouble(params->vals[k]);
        PyDict_SetItemString(py_new_params,params->names[k].c_str(),py_val);
        Py_DECREF(py_val);
    }
    PyObject* py_new_piq = PyList_New((Py_ssize_t)chg_params->Q);
    for (int i = 0; i < chg_params->Q; ++i) {
        PyList_SetItem(py_new_piq, (Py_ssize_t)i,
                PyFloat_FromDouble(exp(chg_params->log_piq[i])));
    }
    PyObject* py_new_qvals = PyList_New((Py_ssize_t)chg_params->Q);
    for (int i = 0; i < chg_params->Q; ++i) {
        PyList_SetItem(py_new_qvals, (Py_ssize_t)i,
                PyFloat_FromDouble(chg_params->q_vals[i]));
    }
    PyObject* py_chg_hist = PyList_New((Py_ssize_t)chg_hist.size());
    for (unsigned i = 0; i < chg_hist.size(); ++i) {
        PyObject* py_chg = PyDict_New();
        for (Changes::const_iterator iter = chg_hist[i].begin();
                iter != chg_hist[i].end(); ++iter) {
            PyObject* py_t = PyLong_FromLong((long)iter->first);
            PyObject* py_js = PyList_New((Py_ssize_t)iter->second.size());
            for (unsigned j = 0; j < iter->second.size(); ++j)
                PyList_SetItem(py_js, (Py_ssize_t)j,
                        PyLong_FromLong((long)iter->second[j]));
            PyDict_SetItem(py_chg, py_t, py_js);
            Py_DECREF(py_t);
            Py_DECREF(py_js);
        }
        PyList_SetItem(py_chg_hist, (Py_ssize_t)i, py_chg);
    }
    PyObject* py_return_tuple = Py_BuildValue("(OOOO)",
            py_chg_hist,py_new_params,py_new_piq,py_new_qvals);
    delete_change_params(chg_params);
    delete_params(params);
    Py_DECREF(np_X);
    Py_DECREF(np_Z);
    Py_DECREF(py_chg_hist);
    Py_DECREF(py_new_params);
    Py_DECREF(py_new_piq);
    Py_DECREF(py_new_qvals);
    return py_return_tuple;
}

/* Posterior mode */
PyObject* py_compute_posterior_mode(PyObject* self, PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_Z = NULL;
    char* model = NULL;
    PyObject* py_params = NULL;
    PyObject* py_q_vals = NULL;
    PyObject* py_piq = NULL;
    int row_time_block = 0;
    int max_iters = 0;
    PyObject* py_col_only = NULL;
    PyObject* py_row_only = NULL;
    PyObject* py_swap = NULL;
    PyObject* py_verbose = NULL;
    if (!PyArg_ParseTuple(args, "OOsOOOiiOOOO", &py_X,&py_Z,&model,
                &py_params,&py_q_vals,&py_piq,&max_iters,&row_time_block,
                &py_col_only,&py_row_only,&py_swap,&py_verbose))
        return NULL;
    bool col_only = (bool)PyObject_IsTrue(py_col_only);
    bool row_only = (bool)PyObject_IsTrue(py_row_only);
    bool swap = (bool)PyObject_IsTrue(py_swap);
    bool verbose = (bool)PyObject_IsTrue(py_verbose);
    ChangeParams* chg_params = NULL;
    if (!py_extract_change_params(py_q_vals, py_piq, chg_params))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params)) {
        delete_change_params(chg_params);
        return NULL;
    }
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject* np_Z = PyArray_FROM_OTF(py_Z, NPY_BOOL, NPY_INOUT_ARRAY);
    int T;
    int J;
    if (!py_check_matrices(np_X, np_Z, T, J)) {
        delete_change_params(chg_params);
        delete_params(params);
        return NULL;
    }
    double* X = (double*) PyArray_DATA(np_X);
    bool* Z = (bool*) PyArray_DATA(np_Z);
    try {
        if (strcmp(model, "normal_mean") == 0) {
            compute_posterior_mode<NormalMeanModel>(X,Z,T,J,params,
                    chg_params,max_iters,row_time_block,col_only,row_only,
                    swap,verbose);
        } else if (strcmp(model, "normal_mean_var") == 0) {
            compute_posterior_mode<NormalMeanVarModel>(X,Z,T,J,params,
                    chg_params,max_iters,row_time_block,col_only,row_only,
                    swap,verbose);
        } else if (strcmp(model, "normal_var") == 0) {
            compute_posterior_mode<NormalVarModel>(X,Z,T,J,params,
                    chg_params,max_iters,row_time_block,col_only,row_only,
                    swap,verbose);
        } else if (strcmp(model, "poisson") == 0) {
            compute_posterior_mode<PoissonModel>(X,Z,T,J,params,
                    chg_params,max_iters,row_time_block,col_only,row_only,
                    swap,verbose);
        } else if (strcmp(model, "bernoulli") == 0) {
            compute_posterior_mode<BernoulliModel>(X,Z,T,J,params,
                    chg_params,max_iters,row_time_block,col_only,row_only,
                    swap,verbose);
        } else if (strcmp(model, "laplace_scale") == 0) {
            compute_posterior_mode<LaplaceScaleModel>(X,Z,T,J,params,
                    chg_params,max_iters,row_time_block,col_only,row_only,
                    swap,verbose);
        } else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_change_params(chg_params);
            delete_params(params);
            Py_DECREF(np_X);
            Py_DECREF(np_Z);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_change_params(chg_params);
        delete_params(params);
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return NULL;
    }
    delete_change_params(chg_params);
    delete_params(params);
    Py_DECREF(np_X);
    Py_DECREF(np_Z);
    Py_INCREF(Py_None);
    return Py_None;
}

/********* Inference subroutines -- expose only for debugging ************/
/*
static PyObject* py_sample_rows(PyObject* self, PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_Z = NULL;
    char* model = NULL;
    PyObject* py_params = NULL;
    PyObject* py_q_vals = NULL;
    PyObject* py_piq = NULL;
    int first_block = -1;
    int block = -1;
    if (!PyArg_ParseTuple(args, "OOsOOOii", &py_X, &py_Z, &model,
                &py_params, &py_q_vals, &py_piq, &first_block, &block))
        return NULL;
    ChangeParams* chg_params = NULL;
    if (!py_extract_change_params(py_q_vals, py_piq, chg_params))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params)) {
        delete_change_params(chg_params);
        return NULL;
    }
    assert(params && chg_params);
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject* np_Z = PyArray_FROM_OTF(py_Z, NPY_BOOL, NPY_INOUT_ARRAY);
    int T;
    int J;
    if (!py_check_matrices(np_X, np_Z, T, J)) {
        delete_change_params(chg_params);
        delete_params(params);
        return NULL;
    }
    chg_params->compute_f_g(J);
    double* X = (double*) PyArray_DATA(np_X);
    bool* Z = (bool*) PyArray_DATA(np_Z);
    try {
        if (strcmp(model, "normal_mean") == 0) {
            sample_rows<NormalMeanModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "normal_mean_var") == 0) {
            sample_rows<NormalMeanVarModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "normal_var") == 0) {
            sample_rows<NormalVarModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "poisson") == 0) {
            sample_rows<PoissonModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "bernoulli") == 0) {
            sample_rows<BernoulliModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "laplace_scale") == 0) {
            sample_rows<LaplaceScaleModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_change_params(chg_params);
            delete_params(params);
            Py_DECREF(np_X);
            Py_DECREF(np_Z);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_change_params(chg_params);
        delete_params(params);
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return NULL;
    }
    delete_change_params(chg_params);
    delete_params(params);
    Py_DECREF(np_X);
    Py_DECREF(np_Z);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* py_sample_columns(PyObject* self, PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_Z = NULL;
    char* model = NULL;
    PyObject* py_params = NULL;
    PyObject* py_q_vals = NULL;
    PyObject* py_piq = NULL;
    double approx = -1;
    if (!PyArg_ParseTuple(args, "OOsOOOd", &py_X, &py_Z, &model,
                &py_params, &py_q_vals, &py_piq, &approx))
        return NULL;
    ChangeParams* chg_params = NULL;
    if (!py_extract_change_params(py_q_vals, py_piq, chg_params))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params)) {
        delete_change_params(chg_params);
        return NULL;
    }
    assert(params && chg_params);
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject* np_Z = PyArray_FROM_OTF(py_Z, NPY_BOOL, NPY_INOUT_ARRAY);
    int T;
    int J;
    if (!py_check_matrices(np_X, np_Z, T, J)) {
        delete_change_params(chg_params);
        delete_params(params);
        return NULL;
    }
    chg_params->compute_f_g(J);
    double* X = (double*) PyArray_DATA(np_X);
    bool* Z = (bool*) PyArray_DATA(np_Z);
    try {
        if (strcmp(model, "normal_mean") == 0) {
            sample_columns<NormalMeanModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "normal_mean_var") == 0) {
            sample_columns<NormalMeanVarModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "normal_var") == 0) {
            sample_columns<NormalVarModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "poisson") == 0) {
            sample_columns<PoissonModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "bernoulli") == 0) {
            sample_columns<BernoulliModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "laplace_scale") == 0) {
            sample_columns<LaplaceScaleModel>(X,Z,T,J,*params,*chg_params);
        } else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_change_params(chg_params);
            delete_params(params);
            Py_DECREF(np_X);
            Py_DECREF(np_Z);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_change_params(chg_params);
        delete_params(params);
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return NULL;
    }
    delete_change_params(chg_params);
    delete_params(params);
    Py_DECREF(np_X);
    Py_DECREF(np_Z);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* py_estimate_parameters(PyObject* self, PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_chg_hist = NULL;
    int MCEM_start = 0;
    char* model = NULL;
    PyObject* py_params = NULL;
    PyObject* py_q_vals = NULL;
    PyObject* py_piq = NULL;
    PyObject* py_estimate_piq = NULL;
    if (!PyArg_ParseTuple(args, "OOisOOOO", &py_X, &py_chg_hist,
                &MCEM_start, &model, &py_params, &py_q_vals, &py_piq,
                &py_estimate_piq))
        return NULL;
    bool estimate_piq = (bool)PyObject_IsTrue(py_estimate_piq);
    std::vector<std::map<int, std::vector<int> > > chg_hist;
    py_extract_chg_hist(py_chg_hist, chg_hist);
    ChangeParams* chg_params = NULL;
    if (!py_extract_change_params(py_q_vals, py_piq, chg_params))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params)) {
        delete_change_params(chg_params);
        return NULL;
    }
    assert(params && chg_params);
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    if (np_X == NULL || (int)PyArray_NDIM(np_X) != 2) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix input");
        delete_change_params(chg_params);
        delete_params(params);
        Py_XDECREF(np_X);
        return NULL;
    }
    int J = (int)PyArray_DIM(np_X, 0);
    int T = (int)PyArray_DIM(np_X, 1);
    chg_params->compute_f_g(J);
    double* X = (double*) PyArray_DATA(np_X);
    try {
        if (strcmp(model, "normal_mean") == 0) {
            estimate_parameters<NormalMeanModel>(X,T,J,chg_hist,
                    MCEM_start,*params,*chg_params,estimate_piq);
        } else if (strcmp(model, "normal_mean_var") == 0) {
            estimate_parameters<NormalMeanVarModel>(X,T,J,chg_hist,
                    MCEM_start,*params,*chg_params,estimate_piq);
        } else if (strcmp(model, "normal_var") == 0) {
            estimate_parameters<NormalVarModel>(X,T,J,chg_hist,
                    MCEM_start,*params,*chg_params,estimate_piq);
        } else if (strcmp(model, "poisson") == 0) {
            estimate_parameters<PoissonModel>(X,T,J,chg_hist,
                    MCEM_start,*params,*chg_params,estimate_piq);
        } else if (strcmp(model, "bernoulli") == 0) {
            estimate_parameters<BernoulliModel>(X,T,J,chg_hist,
                    MCEM_start,*params,*chg_params,estimate_piq);
        } else if (strcmp(model, "laplace_scale") == 0) {
            estimate_parameters<LaplaceScaleModel>(X,T,J,chg_hist,
                    MCEM_start,*params,*chg_params,estimate_piq);
        } else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_change_params(chg_params);
            delete_params(params);
            Py_DECREF(np_X);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_change_params(chg_params);
        delete_params(params);
        Py_DECREF(np_X);
        return NULL;
    }
    PyObject* py_new_params = PyDict_New();
    for (int k = 0; k < params->K; ++k) {
        PyObject* py_val = PyFloat_FromDouble(params->vals[k]);
        PyDict_SetItemString(py_new_params,params->names[k].c_str(),py_val);
        Py_DECREF(py_val);
    }
    PyObject* py_new_piq = PyList_New(chg_params->Q);
    for (int i = 0; i < chg_params->Q; ++i) {
        PyList_SetItem(py_new_piq, i,
                PyFloat_FromDouble(exp(chg_params->log_piq[i])));
    }
    PyObject* py_tuple = Py_BuildValue("(OO)", py_new_params, py_new_piq);
    Py_DECREF(py_new_params);
    Py_DECREF(py_new_piq);
    delete_change_params(chg_params);
    delete_params(params);
    Py_DECREF(np_X);
    return py_tuple;
}

static PyObject* py_metropolis_swap_change_times(PyObject* self,
        PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_Z = NULL;
    char* model = NULL;
    PyObject* py_params = NULL;
    int nswaps = 0;
    if (!PyArg_ParseTuple(args, "OOsOi", &py_X, &py_Z, &model, &py_params,
                &nswaps))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params))
        return NULL;
    assert(params);
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject* np_Z = PyArray_FROM_OTF(py_Z, NPY_BOOL, NPY_INOUT_ARRAY);
    int T;
    int J;
    if (!py_check_matrices(np_X, np_Z, T, J)) {
        delete_params(params);
        return NULL;
    }
    double* X = (double*) PyArray_DATA(np_X);
    bool* Z = (bool*) PyArray_DATA(np_Z);
    try {
        if (strcmp(model, "normal_mean") == 0)
            metropolis_swap_change_times<NormalMeanModel>(X,Z,T,J,*params,
                    nswaps);
        else if (strcmp(model, "normal_mean_var") == 0)
            metropolis_swap_change_times<NormalMeanVarModel>(X,Z,T,J,
                    *params,nswaps);
        else if (strcmp(model, "normal_var") == 0)
            metropolis_swap_change_times<NormalVarModel>(X,Z,T,J,*params,
                    nswaps);
        else if (strcmp(model, "poisson") == 0)
            metropolis_swap_change_times<PoissonModel>(X,Z,T,J,*params,
                    nswaps);
        else if (strcmp(model, "bernoulli") == 0)
            metropolis_swap_change_times<BernoulliModel>(X,Z,T,J,*params,
                    nswaps);
        else if (strcmp(model, "laplace_scale") == 0)
            metropolis_swap_change_times<LaplaceScaleModel>(X,Z,T,J,*params,
                    nswaps);
        else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_params(params);
            Py_DECREF(np_X);
            Py_DECREF(np_Z);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_params(params);
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return NULL;
    }
    delete_params(params);
    Py_DECREF(np_X);
    Py_DECREF(np_Z);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* py_maximize_rows(PyObject* self, PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_Z = NULL;
    char* model = NULL;
    PyObject* py_params = NULL;
    PyObject* py_q_vals = NULL;
    PyObject* py_piq = NULL;
    int first_block = -1;
    int block = -1;
    if (!PyArg_ParseTuple(args, "OOsOOOii", &py_X, &py_Z, &model,
                &py_params, &py_q_vals, &py_piq, &first_block, &block))
        return NULL;
    ChangeParams* chg_params = NULL;
    if (!py_extract_change_params(py_q_vals, py_piq, chg_params))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params)) {
        delete_change_params(chg_params);
        return NULL;
    }
    assert(params && chg_params);
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject* np_Z = PyArray_FROM_OTF(py_Z, NPY_BOOL, NPY_INOUT_ARRAY);
    int T;
    int J;
    if (!py_check_matrices(np_X, np_Z, T, J)) {
        delete_change_params(chg_params);
        delete_params(params);
        return NULL;
    }
    chg_params->compute_f_g(J);
    double* X = (double*) PyArray_DATA(np_X);
    bool* Z = (bool*) PyArray_DATA(np_Z);
    try {
        if (strcmp(model, "normal_mean") == 0) {
            maximize_rows<NormalMeanModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "normal_mean_var") == 0) {
            maximize_rows<NormalMeanVarModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "normal_var") == 0) {
            maximize_rows<NormalVarModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "poisson") == 0) {
            maximize_rows<PoissonModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "bernoulli") == 0) {
            maximize_rows<BernoulliModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else if (strcmp(model, "laplace_scale") == 0) {
            maximize_rows<LaplaceScaleModel>(X,Z,T,J,*params,*chg_params,
                    first_block,block);
        } else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_change_params(chg_params);
            delete_params(params);
            Py_DECREF(np_X);
            Py_DECREF(np_Z);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_change_params(chg_params);
        delete_params(params);
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return NULL;
    }
    delete_change_params(chg_params);
    delete_params(params);
    Py_DECREF(np_X);
    Py_DECREF(np_Z);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* py_maximize_columns(PyObject* self, PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_Z = NULL;
    char* model = NULL;
    PyObject* py_params = NULL;
    PyObject* py_q_vals = NULL;
    PyObject* py_piq = NULL;
    double approx = -1;
    if (!PyArg_ParseTuple(args, "OOsOOOd", &py_X, &py_Z, &model,
                &py_params, &py_q_vals, &py_piq, &approx))
        return NULL;
    ChangeParams* chg_params = NULL;
    if (!py_extract_change_params(py_q_vals, py_piq, chg_params))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params)) {
        delete_change_params(chg_params);
        return NULL;
    }
    assert(params && chg_params);
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject* np_Z = PyArray_FROM_OTF(py_Z, NPY_BOOL, NPY_INOUT_ARRAY);
    int T;
    int J;
    if (!py_check_matrices(np_X, np_Z, T, J)) {
        delete_change_params(chg_params);
        delete_params(params);
        return NULL;
    }
    chg_params->compute_f_g(J);
    double* X = (double*) PyArray_DATA(np_X);
    bool* Z = (bool*) PyArray_DATA(np_Z);
    try {
        if (strcmp(model, "normal_mean") == 0) {
            maximize_columns<NormalMeanModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "normal_mean_var") == 0) {
            maximize_columns<NormalMeanVarModel>(X,Z,T,J,*params,
                    *chg_params);
        } else if (strcmp(model, "normal_var") == 0) {
            maximize_columns<NormalVarModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "poisson") == 0) {
            maximize_columns<PoissonModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "bernoulli") == 0) {
            maximize_columns<BernoulliModel>(X,Z,T,J,*params,*chg_params);
        } else if (strcmp(model, "laplace_scale") == 0) {
            maximize_columns<LaplaceScaleModel>(X,Z,T,J,*params,
                    *chg_params);
        } else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_change_params(chg_params);
            delete_params(params);
            Py_DECREF(np_X);
            Py_DECREF(np_Z);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_change_params(chg_params);
        delete_params(params);
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return NULL;
    }
    delete_change_params(chg_params);
    delete_params(params);
    Py_DECREF(np_X);
    Py_DECREF(np_Z);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* py_greedy_swap_change_times(PyObject* self,
        PyObject* args) {
    PyObject* py_X = NULL;
    PyObject* py_Z = NULL;
    char* model = NULL;
    PyObject* py_params = NULL;
    if (!PyArg_ParseTuple(args, "OOsO", &py_X, &py_Z, &model, &py_params))
        return NULL;
    ModelParams* params = NULL;
    if (!py_extract_params(py_params, params))
        return NULL;
    assert(params);
    PyObject* np_X = PyArray_FROM_OTF(py_X, NPY_FLOAT64, NPY_IN_ARRAY);
    PyObject* np_Z = PyArray_FROM_OTF(py_Z, NPY_BOOL, NPY_INOUT_ARRAY);
    int T;
    int J;
    if (!py_check_matrices(np_X, np_Z, T, J)) {
        delete_params(params);
        return NULL;
    }
    double* X = (double*) PyArray_DATA(np_X);
    bool* Z = (bool*) PyArray_DATA(np_Z);
    try {
        if (strcmp(model, "normal_mean") == 0)
            greedy_swap_change_times<NormalMeanModel>(X,Z,T,J,*params);
        else if (strcmp(model, "normal_mean_var") == 0)
            greedy_swap_change_times<NormalMeanVarModel>(X,Z,T,J,*params);
        else if (strcmp(model, "normal_var") == 0)
            greedy_swap_change_times<NormalVarModel>(X,Z,T,J,*params);
        else if (strcmp(model, "poisson") == 0)
            greedy_swap_change_times<PoissonModel>(X,Z,T,J,*params);
        else if (strcmp(model, "bernoulli") == 0)
            greedy_swap_change_times<BernoulliModel>(X,Z,T,J,*params);
        else if (strcmp(model, "laplace_scale") == 0)
            greedy_swap_change_times<LaplaceScaleModel>(X,Z,T,J,*params);
        else {
            PyErr_SetString(PyExc_ValueError, "Unrecognized model");
            delete_params(params);
            Py_DECREF(np_X);
            Py_DECREF(np_Z);
            return NULL;
        }
    } catch (std::exception& e) {
        std::stringstream ss;
        ss << "Execution error: " << e.what();
        PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
        delete_params(params);
        Py_DECREF(np_X);
        Py_DECREF(np_Z);
        return NULL;
    }
    delete_params(params);
    Py_DECREF(np_X);
    Py_DECREF(np_Z);
    Py_INCREF(Py_None);
    return Py_None;
}
*/

static PyMethodDef methods[] = {
    {"seed", py_seed, METH_VARARGS, "Set seed"},
    {"gibbs_sample", py_gibbs_sample, METH_VARARGS,
        "Gibbs sample changes"},
    {"compute_posterior_mode", py_compute_posterior_mode, METH_VARARGS,
        "Compute posterior mode for changes"},
    /* Inference subroutines -- expose only for debugging
    {"sample_rows", py_sample_rows, METH_VARARGS, "Sample rows"},
    {"sample_columns", py_sample_columns, METH_VARARGS, "Sample columns"},
    {"metropolis_swap_change_times", py_metropolis_swap_change_times,
        METH_VARARGS, "Metropolis swap change times"},
    {"estimate_parameters", py_estimate_parameters, METH_VARARGS,
        "Estimate parameters"},
    {"maximize_rows", py_maximize_rows, METH_VARARGS, "Maximize rows"},
    {"maximize_columns", py_maximize_columns, METH_VARARGS,
        "Maximize columns"},
    {"greedy_swap_change_times", py_greedy_swap_change_times,
        METH_VARARGS, "Greedy swap change times"},
        */
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef _c_funcs_module = {
    // See: https://docs.python.org/3/c-api/module.html
    PyModuleDef_HEAD_INIT, // Basename, it's always this. 
    "_c_funcs",  // Module name
    NULL,        // Optional docstring
    -1,          // Size of per-interpreter state (or -1 for global state)
    methods   // Array of methods
};

// Initialization function
PyMODINIT_FUNC
PyInit__c_funcs(void) {
    // See https://docs.python.org/3/c-api/intro.html#c.PyMODINIT_FUNC
    //
    import_array();      // Initialize NumPy C API
    // We import_array to avoid error with PyArray_FROM_OTF in MCMC sampler.
    // See https://stackoverflow.com/questions/60748039/building-numpy-c-extension-segfault-when-calling-pyarray-from-otf
    
    return PyModule_Create(&_c_funcs_module);
}
