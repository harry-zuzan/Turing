from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

numpy_inc = numpy.get_include()

extensions = [
    Extension("turing.reaction_diffusion",
			["turing/reaction_diffusion.pyx"],
        include_dirs = [numpy_inc],
		)
	]

setup(
    name = "Reaction Diffusion",
    ext_modules = cythonize(extensions),
	packages=['turing'],
)
