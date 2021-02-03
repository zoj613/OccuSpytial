#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
from cpython.pycapsule cimport PyCapsule_GetPointer

import numpy as np
cimport numpy as np
from numpy.random cimport BitGenerator, bitgen_t
from numpy.random.c_distributions cimport random_standard_normal_fill

from scipy.linalg.cython_blas cimport dtrmv
from scipy.linalg.cython_lapack cimport dpotrf, dpotrs


__all__ = ('precision_mvnorm', 'ensure_sums_to_zero')

np.import_array()

cdef const char* FAILURE_MESSAGE = "Cholesky factorization/solver failed!"


def ensure_sums_to_zero(double[::1] x, double[::1] z, double[::1] out):
    """Utility function used when sampling from normal distribution truncated
    on the hyperplace sum(x) = 0.
    """
    cdef Py_ssize_t i, size = out.shape[0]
    cdef double a, x_sum = 0, z_sum = 0

    with nogil:
        for i in range(size):
            x_sum += x[i]
            z_sum += z[i]

        a = - x_sum / z_sum

        for i in range(size):
            out[i] = x[i] + a * z[i]


def precision_mvnorm(double[::1] b, double[:, ::1] prec, random_state=None):
    """
    precision_mvnorm(mean, prec, random_state=None)

    Generate a sample from a gaussian distribution parametrized by its precision.

    The Guassian distribution is of the form

    .. math::

        \mathcal{N}(\mathbf{\Lambda}^{-1}\mathbf{b}, \mathbf{\Lambda}^{-1})

    In most cases only :math:`\mathbf{\Lambda}` and :math:`\mathbf{b}` are
    available. This function facilitates sampling from a Gaussian of this form.

    Parameters
    ----------
    b : np.ndarray
        The :math:`b` component of the distribution's mean.
    prec : np.ndarray
        The precision matrix of the distribution (inverse of the covariance
        matrix). The values are overwitten internally in order to store its
        cholesky factor.
    random_state : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
    A seed to initialize the random number generator. If None, then fresh,
    unpredictable entropy will be pulled from the OS. If an ``int`` or
    ``array_like[ints]`` is passed, then it will be passed to
    `SeedSequence` to derive the initial `BitGenerator` state. One may also
    pass in a `SeedSequence` instance.
    Additionally, when passed a `BitGenerator`, it will be wrapped by
    `Generator`. If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    out : np.ndarray
        A random variable from this distribution.

    """
    cdef BitGenerator bitgenerator
    cdef bitgen_t* bitgen
    cdef np.npy_intp dims
    cdef Py_ssize_t i
    cdef double[::1] out
    cdef double[::1, :] chol
    cdef int info = 0, n = b.shape[0], incx = 1

    bitgenerator = np.random.default_rng(random_state)._bit_generator
    bitgen = <bitgen_t*>PyCapsule_GetPointer(bitgenerator.capsule, "BitGenerator")

    chol = prec.T  # change the precision to Fortran order for use with LAPACK.
    dims = <np.npy_intp>(b.shape[0])
    out = np.PyArray_EMPTY(1, &dims, np.NPY_DOUBLE, 1)

    with bitgenerator.lock, nogil:
        random_standard_normal_fill(bitgen, <np.npy_intp>n, &out[0])

    with nogil:
        # LAPACK cholesky decomposition
        dpotrf('U', &n, &chol[0, 0], &n, &info)
        # BLAS matrix-vector product
        dtrmv('U', 'T', 'N', &n, &chol[0, 0], &n, &out[0], &incx)
        for i in range(n):
            out[i] += b[i]
        dpotrs('U', &n, &incx, &chol[0, 0], &n, &out[0], &n, &info)

    if info:
        raise RuntimeError(FAILURE_MESSAGE)

    return out.base
