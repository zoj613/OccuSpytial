from distutils.command.build_ext import build_ext
from distutils.core import Extension

import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        "occuspytial/**", ["occuspytial/**.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[np.get_include()],
    ),
    Extension(
        "occuspytial/gibbs/*", ["occuspytial/gibbs/*.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[np.get_include()],
    ),
]

ext = cythonize(extensions, include_path=[np.get_include()], annotate=True)


# source: https://github.com/sdispater/pendulum/blob/1.x/build.py
def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update({
        'ext_modules': ext,
        'cmdclass': {'build_ext': build_ext}
    })
