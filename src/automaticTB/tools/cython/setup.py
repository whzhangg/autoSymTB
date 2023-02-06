from distutils.core import setup
from Cython.Build import cythonize
import os, numpy

"""run `python setup.py build_ext --inplace` """

os.environ["CC"] = "g++-12"
setup(
	ext_modules=cythonize("cpivoting.pyx", compiler_directives={"language_level": "3"}),
	include_dirs=[numpy.get_include()]
)

