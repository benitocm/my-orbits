from distutils.core import Extension, setup
from distutils.extension import Extension
from Cython.Build import cythonize

# The instructions to build the cython are :
#       python setup.py build_ext --inplace
#   

ext_modules = [
    Extension (
        r'kepler_u', [r'kepler_u.pyx']
    )
]

setup (
    name='kepler_u',
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level' : "3"})    
)
