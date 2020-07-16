from distutils.core import Extension
from os.path import join

import numpy as np
from Cython.Build import cythonize, build_ext


# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
extensions = [
    Extension(
        "occuspytial.distributions",
        ["occuspytial/distributions.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[join(np.get_include(), '..', '..', 'random', 'lib')],
        libraries=['npyrandom'],
        define_macros=[('NPY_NO_DEPRECATED_API', 0)],
    ),
    Extension(
        "occuspytial.data",
        ["occuspytial/data.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[np.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 0)],
    ),
]


# source: https://github.com/sdispater/pendulum/blob/1.x/build.py
def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update({
        'ext_modules': cythonize(
            extensions,
            compiler_directives={'embedsignature': True}
        ),
        'cmdclass': {'build_ext': build_ext}
    })
