"""This module is a helper script that reads in a C shared library that
is compiled automatically during installation of this package. The lin-
rary helps interface the C function called "_prod" with Python and said
function is used to speed up computation during posterior sampling when
updating the posterior parameter values of z (occupancy state of each
site).

"""
import ctypes
from pathlib import Path
import sys

from numpy.ctypeslib import ndpointer

# store current directory
_currentdir = Path(__file__).parent
_platform = sys.platform  # get the _platform's name
if _platform == "linux":
    _sharedlib_path = next(_currentdir.glob("*.so"))
elif _platform == "darwin":
    _sharedlib_path = next(_currentdir.glob("*.dylib"))
elif _platform in ["win32", "cygwin"]:
    _sharedlib_path = next(_currentdir.glob("*.dll"))
else:
    raise Exception("Platform is not supported.")

_lib = ctypes.cdll.LoadLibrary(_sharedlib_path.as_posix())
occu_prob = _lib._proba
occu_prob.restype = None
occu_prob.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ndpointer(ctypes.c_size_t, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
]
