import ctypes
from pathlib import Path
import sys

#from numpy.ctypeslib import ndpointer

# store current directory
_currentdir = Path(__file__).resolve().parent
platform = sys.platform
if platform == "linux":
    _sharedlib_path = _currentdir / "ctypes_helper.so"
elif platform == "darwin":
    _sharedlib_path = _currentdir / "ctypes_helper.dylib"
elif platform in ["win32", "cygwin"]:
    _sharedlib_path = _currentdir / "ctypes_helper.dll"
else:
    raise Exception("Platform is not supported.")

_lib = ctypes.cdll.LoadLibrary(_sharedlib_path.as_posix())
num_prod = _lib._prod
num_prod.restype = None
num_prod.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ndpointer(ctypes.c_long, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
]
