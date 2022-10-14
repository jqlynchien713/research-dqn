from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("apply_bert.pyx"),
    include_dirs=[numpy.get_include()]
)    