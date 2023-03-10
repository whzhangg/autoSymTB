from distutils.core import setup
from Cython.Build import cythonize
import os, numpy, shutil

"""run `python setup.py build_ext --inplace` """

for c_compiler in ['g++-12', 'g++-11', 'icpc', 'gcc']:
	if shutil.which(c_compiler) is not None:
		compiler = c_compiler
		break

os.environ["CC"] = compiler
setup(
	ext_modules=cythonize("cython_functions.pyx", compiler_directives={"language_level": "3"}),
	include_dirs=[numpy.get_include()]
)

