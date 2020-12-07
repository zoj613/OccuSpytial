from distutils.core import Extension
from os.path import join

import numpy as np


include_dirs = [np.get_include()]
macros = [('NPY_NO_DEPRECATED_API', 0)]

# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
extensions = [
    Extension(
        "occuspytial.distributions",
        ["occuspytial/distributions.c"],
        include_dirs=include_dirs,
        library_dirs=[join(np.get_include(), '..', '..', 'random', 'lib')],
        libraries=['npyrandom'],
        define_macros=macros,
    ),
    Extension(
        "occuspytial.data",
        ["occuspytial/data.c"],
        include_dirs=include_dirs,
        define_macros=macros,
    ),
]


def build(setup_kwargs):
    """Build extension modules."""
    setup_kwargs.update(ext_modules=extensions)
