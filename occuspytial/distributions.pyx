#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
from numbers import Number
import warnings

import numpy as np
from pypolyagamma import PyPolyaGamma
from scipy.linalg.cython_blas cimport dtrmv, dtrsv
from scipy.linalg.cython_lapack cimport dpotrf
from scipy.sparse.linalg import splu, cg
try:
    from sksparse.cholmod import cholesky as sparse_cholesky
    USE_SKSPARSE = True
except ImportError:
    warnings.warn(
        'scikit sparse is not installed. Inference may be slow. '
        'To ensure maximum speed during sampling, please install the '
        'sckit-sparse package via "pip install sckit-sparse".'
    )
    USE_SKSPARSE = False

__all__ = (
    'SumToZeroMultivariateNormal',
    'SparseMultivariateNormal',
    'DenseMultivariateNormal',
    'PolyaGamma',
)

cdef dict OPTS = {'SymmetricMode': True}


cdef class Distribution:
    cdef object seed
    cdef object rng

    def __cinit__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __reduce__(self):
        return self.__class__, (self.seed,)


cdef class SparseMultivariateNormal(Distribution):

    cpdef tuple rvs(self, double[:] mean, cov):
        cdef int size = mean.shape[0]

        factor = sparse_cholesky(cov, ordering_method='default')
        L = factor.L()
        chol = factor.apply_Pt(L)
        arr = self.rng.standard_normal(size)
        arr = mean + chol @ arr
        return arr, factor


cdef class DenseMultivariateNormal(Distribution):

    cpdef tuple rvs(self, double[:] mean, double[::1, :] cov):
        cdef int n = mean.shape[0]
        cdef Py_ssize_t i
        cdef int info, incx = 1

        with nogil:
            # LAPACK cholesky decomposition
            dpotrf('U', &n, &cov[0, 0], &n, &info)

        if info != 0:
            raise RuntimeError('Cholesky Factorization failed')

        std = self.rng.standard_normal(n)

        cdef double[::1] std_v = std

        with nogil:
            # BLAS matrix-vector product
            dtrmv('U', 'T', 'N', &n, &cov[0, 0], &n, &std_v[0], &incx)
            for i in range(n):
                std_v[i] += mean[i]

        return std, cov



cdef class DenseMultivariateNormal2(Distribution):
    
    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, seed=None):
        self.mvnorm = DenseMultivariateNormal(seed)

    def rvs(self, double[:] b, double[:, :] prec):
        cdef int n = b.shape[0]
        cdef int incx = 1
        cdef double[::1, :] prec_d = np.asfortranarray(prec)

        s, chol = self.mvnorm.rvs(b, prec_d)
        cdef double[:] s_v = s
        cdef double[::1, :] chol_v = chol

        with nogil:
            # BLAS triangular matrix direct solver
            dtrsv('U', 'T', 'N', &n, &chol_v[0, 0], &n, &s_v[0], &incx)
            dtrsv('U', 'N', 'N', &n, &chol_v[0, 0], &n, &s_v[0], &incx)

        return s


cdef void scale_arr(double[:] x, double[:] z, double[:] out, int size) nogil:
    cdef Py_ssize_t i
    cdef double x_sum = 0, z_sum = 0
    cdef double a
    # a = -x.sum() / z.sum()
    for i in range(size):
        x_sum += x[i]
        z_sum += z[i]
    a = - x_sum / z_sum
    # out = x + a * z
    for i in range(size):
        out[i] = x[i] + a * z[i]


cdef class FastSumToZeroMultivariateNormal(Distribution):

    cdef SparseMultivariateNormal mvnorm

    def __cinit__(self, seed=None):
        self.mvnorm = SparseMultivariateNormal(seed)

    def rvs(self, double[:] b, prec):
        cdef int size = b.shape[0]
        cdef double[:] x, z

        s, factor = self.mvnorm.rvs(b, prec)
        cdef double[:] s_v = s

        x = factor.solve_A(s)
        s_v[...] = 1 
        z = factor.solve_A(s)

        scale_arr(x, z, s_v, size)
        return s


cdef class SlowSumToZeroMultivariateNormal(Distribution):

    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, seed=None):
        self.mvnorm = DenseMultivariateNormal(seed)

    def rvs(self, double[:] b, prec):
        cdef int n = b.shape[0]
        cdef int incx = 1
        cdef double[::1, :] prec_d = prec.toarray(order='F')

        s, chol = self.mvnorm.rvs(b, prec_d)
        cdef double[::1] x = s
        cdef double[::1, :] chol_v = chol
        cdef double[::1] z = prec_d[:, 0]
        z[...] = 1

        with nogil:
            dtrsv('U', 'T', 'N', &n, &chol_v[0, 0], &n, &x[0], &incx)
            dtrsv('U', 'N', 'N', &n, &chol_v[0, 0], &n, &x[0], &incx)

            dtrsv('U', 'T', 'N', &n, &chol_v[0, 0], &n, &z[0], &incx)
            dtrsv('U', 'N', 'N', &n, &chol_v[0, 0], &n, &z[0], &incx)

        scale_arr(x, z, x, n)
        return s


cdef class SlowSumToZeroMultivariateNormal2(Distribution):

    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, seed=None):
        self.mvnorm = DenseMultivariateNormal(seed)

    def rvs(self, double[:] b, prec):
        superlu = splu(prec, permc_spec='MMD_AT_PLUS_A', options=OPTS)
        cdef int size = b.shape[0]
        cdef double[:, :] xz
        cdef double[:] x, z
        
        prec_d = prec.toarray(order='F')
        cdef double[::1, :] prec_d_v = prec_d
        s, _ = self.mvnorm.rvs(b, prec_d_v)
        cdef double[:] s_v = s

        prec_d_v[:, 0] = s_v
        prec_d_v[:, 1] = 1
        rhs = prec_d[:, :2]
        xz = superlu.solve(rhs)
        x = xz[:, 0]
        z = xz[:, 1]

        scale_arr(x, z, s_v, size)
        return s


cdef class SlowSumToZeroMultivariateNormal3(Distribution):

    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, seed=None):
        self.mvnorm = DenseMultivariateNormal(seed)

    def rvs(self, double[:] b, prec, precond=None):
        cdef int size = b.shape[0]
        cdef double[:] x, z

        prec_d = prec.toarray(order='F')
        cdef double[::1, :] prec_d_v = prec_d
        s, _ = self.mvnorm.rvs(b, prec_d_v)
        cdef double[:] s_v = s

        prec_d_v[:, 0] = 1
        x = cg(prec, s, M=precond)[0]
        z = cg(prec, prec_d[:, 0], M=precond)[0]

        scale_arr(x, z, s_v, size)
        return s


if USE_SKSPARSE:
    SumToZeroMultivariateNormal = FastSumToZeroMultivariateNormal
else:
    SumToZeroMultivariateNormal = SlowSumToZeroMultivariateNormal


cdef class PolyaGamma(Distribution):

    def __cinit__(self, seed=None):
        if seed is None:
            seed = self.rng.integers(low=0, high=2 ** 63)
        self.rng = PyPolyaGamma(seed)

    def rvs(self, a, b, double[:] out=None):
        if isinstance(a, Number):
            return self.rng.pgdraw(a, b)
        elif out is not None:
            self.rng.pgdrawv(a, b, out)
        else:
            out_arr = np.zeros(b.shape[0], dtype=np.double)
            self.rng.pgdrawv(a, b, out_arr)
            return out_arr
