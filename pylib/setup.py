from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
import numpy

exec(open('BASIC_changepoint/_version.py').read())

setup(name="BASIC_changepoint",
        version=__version__,
        description="BASIC (Bayesian Analysis of SImultaneous Changepoints)",
        author="Zhou Fan",
        packages=['BASIC_changepoint'],
        ext_modules=[Extension('BASIC_changepoint._c_funcs',
            sources=['BASIC_changepoint/py_extension.cpp',
                '../src/base.cpp',
                '../src/inference_procedures.cpp',
                '../src/bernoulli_model.cpp',
                '../src/laplace_scale_model.cpp',
                '../src/normal_mean_model.cpp',
                '../src/normal_mean_var_model.cpp',
                '../src/normal_var_model.cpp',
                '../src/poisson_model.cpp'],
            include_dirs=[get_python_inc(), numpy.get_include()],
            extra_compile_args=['-std=c++11'])]
        )
