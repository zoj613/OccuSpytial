#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
from numbers import Number

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


__all__ = (
    'DenseMultivariateNormal',
    'DenseMultivariateNormal2',
    'PolyaGamma',
)

np.import_array()

cdef const char* CHOLESKY_FAILURE = 'Cholesky factorization failed!'


cdef class Distribution:
    """Base class to serve as an informal interface for distributions.

    Implementations can inherit from this class, so it should never be used
    directly.
    """

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

    def __reduce__(self):
        return self.__class__, (self.random_state,)


cdef class DenseMultivariateNormal(Distribution):
    """Multivariate Gaussian distribution for dense covariance matrices.

    Parameters
    ----------
    random_state : {None, int, numpy.random.SeedSequence}
        A seed to initialize the random number generator. Defaults to None.

    Methods
    -------
    rvs(mean, cov, overwrite_cov=True)
    """

    cdef inline void crvs(self, double* mean, double* cov, double* out, int* info, int* n) nogil:
        cdef:
            Py_ssize_t i
            int incx = 1

        # LAPACK cholesky decomposition
        dpotrf('U', n, cov, n, info)
        random_standard_normal_fill(self.rng, <np.npy_intp>n[0], out)
        # BLAS matrix-vector product
        dtrmv('U', 'T', 'N', n, cov, n, out, &incx)
        for i in range(n[0]):
            out[i] += mean[i]

    def rvs(self, double[::1] mean, double[:, ::1] cov, bint overwrite_cov=True):
        """
        rvs(mean, cov, overwrite_cov=True)

        Generate a random sample from a multivariate Gaussian distribution.
        
        Parameters
        ----------
        mean : np.ndarray
            Mean of the distribution
        cov : np.ndarray
            Covariance matrix of the distribution.
        overwrite_cov : bool
            Whether to write over the `cov` array when computing the cholesky
            factor instead of creating a new array. Defaults to True.

        Returns
        -------
        out : np.ndarray
            A random sample from a multivariate Gaussian with mean vector
            and covariance specified by `mean` and `cov`.
        factor : np.ndarray
            The cholesky factor of the covariance matrix :math:`\mathbf{U}`
            such that :math:`\mathbf{U^TU} = \mathbf{A}`. All the data is in
            the upper triangular part of the array, The lower triangular part
            of the array is garbage values.

        """
        cdef:
            np.npy_intp* dims = <np.npy_intp*>(mean.shape)
            out = np.PyArray_EMPTY(1, dims, np.NPY_DOUBLE, 1)
            double[::1] out_v = out
            double[::1, :] chol
            int n = out_v.shape[0]
            int info

        if not overwrite_cov:
            a = np.PyArray_NewCopy(<np.ndarray>cov.base, np.NPY_FORTRANORDER)
            chol = a
        else:
            chol = cov.T

        self.crvs(&mean[0], &chol[0, 0], &out_v[0], &info, &n)

        if info:
            raise RuntimeError(CHOLESKY_FAILURE)

        return out


cdef class DenseMultivariateNormal2(Distribution):
    """Multivariate Gaussian distribution for dense precision matrices.

    The Guassian distribution is of the form

    .. math::

        \mathcal{N}(\mathbf{\Lambda}^{-1}\mathbf{b}, \mathbf{\Lambda}^{-1})

    In most cases only :math:`\mathbf{\Lambda}` and :math:`\mathbf{b}` are
    available. This class facilitates sampling from a Gaussian of this form.

    Parameters
    ----------
    random_state : {None, int, numpy.random.SeedSequence}
        A seed to initialize the random number generator. Defaults to None.

    Methods
    -------
    rvs(mean, prec, overwrite_prec=True)
    """

    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

    def rvs(self, double[::1] b, double[:, ::1] prec, bint overwrite_prec=True):
        """
        rvs(mean, prec, overwite_prec=True)

        Generate a random draw from this distribution.

        Parameters
        ----------
        b : np.ndarray
            The :math:`b` component of the distribution's mean.
        prec : np.ndarray
            The precision matrix of the distribution (inverse of the covariance
            matrix).
        overwrite_cov : bool
            Whether to write over the `prec` array when computing the cholesky
            factor instead of creating a new array. Defaults to True.

        Returns
        -------
        out : np.ndarray
            A random variable from this distribution.

        """
        cdef:
            np.npy_intp* dims = <np.npy_intp*>(b.shape)
            out = np.PyArray_EMPTY(1, dims, np.NPY_DOUBLE, 1)
            double[::1] out_v = out
            double[::1, :] chol
            int n = b.shape[0]
            int incx = 1
            int info

        if not overwrite_prec:
            a = np.PyArray_NewCopy(<np.ndarray>prec.base, np.NPY_FORTRANORDER)
            chol = a
        else:
            chol = prec.T

        self.mvnorm.crvs(&b[0], &chol[0, 0], &out_v[0], &info, &n)

        if info:
            raise RuntimeError(CHOLESKY_FAILURE)

        with nogil:
            # BLAS triangular matrix direct solver
            dtrsv('U', 'T', 'N', &n, &chol[0, 0], &n, &out_v[0], &incx)
            dtrsv('U', 'N', 'N', &n, &chol[0, 0], &n, &out_v[0], &incx)

        return out


def ensure_sums_to_zero(
    double[::1] x, double[::1] z, double[::1] out, int size
):
    """Utility function used when sampling from normal distribution truncated
    on the hyperplace sum(x) = 0.
    """
    cdef:
        Py_ssize_t i
        double x_sum = 0, z_sum = 0
        double a

    with nogil:
        for i in range(size):
            x_sum += x[i]
            z_sum += z[i]

        a = - x_sum / z_sum

        for i in range(size):
            out[i] = x[i] + a * z[i]


cdef class PolyaGamma(Distribution):
    """Polyagamma distribution random sampler.

    This class is facilitates random sampling from a polya-gamma distribution

    .. math:: PG(b,z)

    The implementation wrapped by this class is the one described in [1]_

    Parameters
    ----------
    random_state : {None, int}
        A seed to initialize the random number generator. Defaults to None.

    Methods
    -------
    rvs(b, z, out)

    References
    ----------
    .. [1]  Windle, J., Polson, N. G., Scott, J. G.,. Sampling Polya-Gamma
       random variates: alternate and approximate techniques (2014). arXiv 
       e-prints arXiv:1405.0506.
    """
    cdef object rng_pg

    def __cinit__(self, random_state=None):
        cdef long random_int = 0
        if random_state is None:
            with nogil:
                random_int = random_positive_int64(self.rng)
        self.rng_pg = PyPolyaGamma(random_int)

    def rvs_arr(self, double[::1] b, double[::1] z, double[::1] out):
        """
        rvs_arr(b, z, out)

        Same as `rvs` but only accepts array input.

        This function is convenient for efficient access when an ouput array
        is provided.
        """
        self.rng_pg.pgdrawv(b, z, out)

    def rvs(self, b, z, double[:] out=None):
        """
        rvs(b, z, out)

        Sample a random draw from a Polya-gamma distribution with parameters

        Parameters
        ----------
        b : {np.ndarray, number}
            The "b" parameter of the :math:`PG(b,z)` distribution.
        z : {np.ndarray, number}
            The "z" parameter of the :math:`PG(b,z)` distribution.
        out : {None, np.ndarray}, optional
            If provided, the output will be assign to this parameter. This
            only applies if both `b` and `z` are arrays. Defaults to None.

        Returns
        -------
        out : {float, np.ndarray}
            Polya-gamma random draw.
        """
        cdef np.npy_intp* dims

        if isinstance(b, Number):
            return self.rng_pg.pgdraw(b, z)
        elif out is not None:
            self.rvs_arr(b, z, out)
        else:
            dims = np.PyArray_DIMS(b)
            out_arr = np.PyArray_EMPTY(1, dims, np.NPY_DOUBLE, 0)
            self.rng_pg.pgdrawv(b, z, out_arr)
            return out_arr
