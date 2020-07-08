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
from numpy.random.c_distributions cimport (
    random_standard_normal_fill, random_positive_int64
)

from pypolyagamma import PyPolyaGamma
from scipy.linalg.cython_blas cimport dtrmv, dtrsv
from scipy.linalg.cython_lapack cimport dpotrf
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

    cpdef tuple rvs(self, np.ndarray mean, cov):
        cdef:
            np.npy_intp* dims = np.PyArray_DIMS(mean)
            np.npy_intp size = np.PyArray_SIZE(mean)
            np.ndarray out = np.PyArray_EMPTY(1, dims, np.NPY_DOUBLE, 0)

        factor = sparse_cholesky(cov, ordering_method='default')
        L = factor.L()
        chol = factor.apply_Pt(L)

        with self.lock, nogil:
            array_data = <double*>np.PyArray_DATA(out)
            random_standard_normal_fill(self.rng, size, array_data)

        out = mean + chol * out
        return out, factor


cdef class DenseMultivariateNormal(Distribution):

    cpdef tuple rvs(self, np.ndarray mean, cov, bint overwrite_cov=True):
        cdef:
            Py_ssize_t i
            np.ndarray chol_arr
            np.npy_intp* dims = np.PyArray_DIMS(mean)
            np.ndarray out = np.PyArray_EMPTY(1, dims, np.NPY_DOUBLE, 1)
            int info, incx = 1
            double[::1, :] chol = cov
            double[::1] out_v = out
            double[::1] mean_v = mean
            int n = mean_v.shape[0]
            np.npy_intp size = <np.npy_intp>n
            double* array_data

        if not overwrite_cov:
            chol_arr = np.PyArray_NewCopy(cov, np.NPY_FORTRANORDER)
            chol = chol_arr

        with nogil:
            # LAPACK cholesky decomposition
            dpotrf('U', &n, &chol[0, 0], &n, &info)

        if info != 0:
            raise RuntimeError('Cholesky Factorization failed')

        with self.lock, nogil:
            array_data = <double*>np.PyArray_DATA(out)
            random_standard_normal_fill(self.rng, size, array_data)

        with nogil:
            # BLAS matrix-vector product
            dtrmv('U', 'T', 'N', &n, &chol[0, 0], &n, &out_v[0], &incx)
            for i in range(n):
                out_v[i] += mean_v[i]

        factor = np.PyArray_FROM_O(chol)
        return out, factor


cdef class DenseMultivariateNormal2(Distribution):
    
    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

    def rvs(self, np.ndarray b, prec):
        cdef:
            int n = <int>np.PyArray_SIZE(b)
            int incx = 1
            double[:] out_v
            double[::1, :] chol_v

        prec_d = np.PyArray_FROM_OF(prec, np.NPY_ARRAY_F_CONTIGUOUS)
        out, chol = self.mvnorm.rvs(b, prec_d)
        out_v = out
        chol_v = chol

        with nogil:
            # BLAS triangular matrix direct solver
            dtrsv('U', 'T', 'N', &n, &chol_v[0, 0], &n, &out_v[0], &incx)
            dtrsv('U', 'N', 'N', &n, &chol_v[0, 0], &n, &out_v[0], &incx)

        return out


cdef void scale_arr(double[:] x, double[:] z, double[:] out, int size) nogil:
    cdef:
        Py_ssize_t i
        double x_sum = 0, z_sum = 0
        double a

    for i in range(size):
        x_sum += x[i]
        z_sum += z[i]

    a = - x_sum / z_sum

    for i in range(size):
        out[i] = x[i] + a * z[i]


cdef class FastSumToZeroMultivariateNormal(Distribution):

    cdef SparseMultivariateNormal mvnorm

    def __cinit__(self, random_state=None):
        self.mvnorm = SparseMultivariateNormal(random_state)

    def rvs(self, np.ndarray b, prec):
        cdef int size = <int>np.PyArray_SIZE(b)
        cdef double[:] x, z, out_v

        out, factor = self.mvnorm.rvs(b, prec)
        out_v = out

        x = factor.solve_A(out)
        out_v[...] = 1 
        z = factor.solve_A(out)

        scale_arr(x, z, out_v, size)
        return out


cdef class SlowSumToZeroMultivariateNormal(Distribution):

    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

    def rvs(self, np.ndarray b, prec):
        cdef:
            int n = <int>np.PyArray_SIZE(b)
            int incx = 1
            double[::1, :] prec_v, chol_v
            double[::1] x, z


        prec_d = prec.toarray(order='F')
        prec_v = prec_d

        out, chol = self.mvnorm.rvs(b, prec_d)
        x = out
        chol_v = chol
        z = prec_v[:, 0]
        z[...] = 1

        with nogil:
            dtrsv('U', 'T', 'N', &n, &chol_v[0, 0], &n, &x[0], &incx)
            dtrsv('U', 'N', 'N', &n, &chol_v[0, 0], &n, &x[0], &incx)

            dtrsv('U', 'T', 'N', &n, &chol_v[0, 0], &n, &z[0], &incx)
            dtrsv('U', 'N', 'N', &n, &chol_v[0, 0], &n, &z[0], &incx)

        scale_arr(x, z, x, n)
        return out


if USE_SKSPARSE:
    SumToZeroMultivariateNormal = FastSumToZeroMultivariateNormal
else:
    SumToZeroMultivariateNormal = SlowSumToZeroMultivariateNormal


cdef class PolyaGamma(Distribution):
    cdef object rng_pg

    def __cinit__(self, random_state=None):
        cdef long random_int = 0
        if random_state is None:
            with self.lock, nogil:
                random_int = random_positive_int64(self.rng)
        self.rng_pg = PyPolyaGamma(random_int)

    def rvs(self, a, b, double[:] out=None):
        cdef np.npy_intp* dims

        if isinstance(a, Number):
            return self.rng_pg.pgdraw(a, b)
        elif out is not None:
            self.rng_pg.pgdrawv(a, b, out)
        else:
            dims = np.PyArray_DIMS(a)
            out_arr = np.PyArray_EMPTY(1, dims, np.NPY_DOUBLE, 0)
            self.rng_pg.pgdrawv(a, b, out_arr)
            return out_arr
