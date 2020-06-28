#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
from numbers import Number
import warnings

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
import numpy as np
cimport numpy as np
from numpy.random cimport bitgen_t
from numpy.random.c_distributions cimport random_standard_normal_fill

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

np.import_array()

cdef dict OPTS = {'SymmetricMode': True}


cdef class Distribution:
    cdef object random_state
    cdef object bitgen
    cdef bitgen_t* rng
    cdef object lock

    def __cinit__(self, random_state=None):
        self.bitgen = np.random.SFC64(random_state)
        cdef const char* capsule_name = 'BitGenerator'
        capsule = self.bitgen.capsule

        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")

        self.rng = <bitgen_t*>PyCapsule_GetPointer(capsule, capsule_name)
        self.random_state = random_state
        self.lock = self.bitgen.lock

    def __reduce__(self):
        return self.__class__, (self.random_state,)


cdef class SparseMultivariateNormal(Distribution):

    cpdef tuple rvs(self, double[:] mean, cov):
        cdef int size = mean.shape[0]
        cdef np.ndarray arr = np.empty(size)

        factor = sparse_cholesky(cov, ordering_method='default')
        L = factor.L()
        chol = factor.apply_Pt(L)

        with self.lock, nogil:
            random_standard_normal_fill(
                self.rng, size, <double*>np.PyArray_DATA(arr)
            )

        arr = mean + chol * arr
        return arr, factor


cdef class DenseMultivariateNormal(Distribution):

    cpdef tuple rvs(self, double[:] mean, double[::1, :] cov, bint overwrite_cov=True):
        cdef int n = mean.shape[0]
        cdef Py_ssize_t i
        cdef int info, incx = 1
        cdef double[::1, :] chol = cov
        cdef np.ndarray std = np.empty(n)

        if not overwrite_cov:
            chol = np.copy(cov, order='F')

        with nogil:
            # LAPACK cholesky decomposition
            dpotrf('U', &n, &chol[0, 0], &n, &info)

        if info != 0:
            raise RuntimeError('Cholesky Factorization failed')

        with self.lock, nogil:
            random_standard_normal_fill(
                self.rng, n, <double*>np.PyArray_DATA(std)
            )

        cdef double[::1] std_v = std

        with nogil:
            # BLAS matrix-vector product
            dtrmv('U', 'T', 'N', &n, &chol[0, 0], &n, &std_v[0], &incx)
            for i in range(n):
                std_v[i] += mean[i]

        return std, chol



cdef class DenseMultivariateNormal2(Distribution):
    
    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

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

    def __cinit__(self, random_state=None):
        self.mvnorm = SparseMultivariateNormal(random_state)

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

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

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

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

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

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

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


cdef class PolyaGamma:
    cdef object random_state
    cdef object rng

    def __cinit__(self, random_state=None):
        if random_state is None:
            rng = np.random.default_rng(np.random.SFC64(random_state))
            random_state = rng.integers(low=0, high=2 ** 63)
        self.rng = PyPolyaGamma(random_state)
        self.random_state = random_state

    def __reduce__(self):
        return self.__class__, (self.random_state,)

    def rvs(self, a, b, double[:] out=None):
        if isinstance(a, Number):
            return self.rng.pgdraw(a, b)
        elif out is not None:
            self.rng.pgdrawv(a, b, out)
        else:
            out_arr = np.zeros(b.shape[0], dtype=np.double)
            self.rng.pgdrawv(a, b, out_arr)
            return out_arr
