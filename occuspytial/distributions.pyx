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
        self.lock = self.bitgen.lock

    def __reduce__(self):
        return self.__class__, (self.random_state,)


cdef class SparseMultivariateNormal(Distribution):
    """Multivariate Gaussian distribution for sparse covariance matrices.

    Parameters
    ----------
    random_state : {None, int, numpy.random.SeedSequence}
        A seed to initialize the random number generator. Defaults to None.

    Methods
    -------
    rvs(mean, cov)
    """
    cpdef tuple rvs(self, np.ndarray mean, cov):
        """
        rvs(self, mean, cov)

        Generate a random draw from this distribution.
        
        Parameters
        ----------
        mean : np.ndarray
            Mean of the distribution
        cov : scipy's sparse matrix format
            Covariance matrix of the distribution.

        Returns
        -------
        out : np.ndarray
            A random sample from a multivariate Gaussian with mean vector
            and covariance specified by `mean` and `cov`.
        factor : sksparse.cholmod.Factor
            The cholesky factor object of the covariance matrix.

        """
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
    """Multivariate Gaussian distribution for dense covariance matrices.

    Parameters
    ----------
    random_state : {None, int, numpy.random.SeedSequence}
        A seed to initialize the random number generator. Defaults to None.

    Methods
    -------
    rvs(mean, cov, overwrite_cov=True)
    """
    cpdef tuple rvs(self, np.ndarray mean, cov, bint overwrite_cov=True):
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
    rvs(mean, prec)
    """
    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

    def rvs(self, np.ndarray b, prec):
        """
        rvs(mean, prec)

        Generate a random draw from this distribution.

        Parameters
        ----------
        b : np.ndarray
            The :math:`b` component of the distribution's mean.
        prec : np.ndarray
            The precision matrix of the distribution (inverse of the covariance
            matrix).

        Returns
        -------
        out : np.ndarray
            A random variable from this distribution.

        """
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
    """Multivariate Gaussian distribution truncated on a hyperplane.

    This class represents a multivariate Gaussian of the form:

    .. math::

        \mathcal{N}(\mathbf{\Lambda}^{-1}\mathbf{b}, \mathbf{\Lambda}^{-1}),

    truncated on the hyperplane :math:`\mathbf{1}^T\mathbf{x} = \mathbf{0}`

    In most cases only :math:`\mathbf{\Lambda}` and :math:`\mathbf{b}` are
    available. This class facilitates sampling from a Gaussian of this form.

    Parameters
    ----------
    random_state : {None, int, numpy.random.SeedSequence}
        A seed to initialize the random number generator. Defaults to None.

    Methods
    -------
    rvs(mean, prec)
    """  
    cdef SparseMultivariateNormal mvnorm

    def __cinit__(self, random_state=None):
        self.mvnorm = SparseMultivariateNormal(random_state)

    def rvs(self, np.ndarray b, prec):
        """
        rvs(mean, prec)

        Generate a random draw from this distribution.

        Parameters
        ----------
        b : np.ndarray
            The :math:`b` component of the distribution's mean.
        prec : np.ndarray
            The precision matrix of the distribution (inverse of the covariance
            matrix).

        Returns
        -------
        out : np.ndarray
            A random variable from this distribution.

        Notes
        -----
        The algorithm implemented here is Algorithm 2 of [1]_ where the
        :math:`\mathbf{G}` in our case is :math:`\mathbf{1}^T` and
        :math:`\mathbf{r}` is :math:`\mathbf{0}`.

        References
        ----------
        .. [1] Cong, Yulai; Chen, Bo; Zhou, Mingyuan. Fast Simulation of 
           Hyperplane-Truncated Multivariate Normal Distributions. Bayesian 
           Anal. 12 (2017), no. 4, 1017--1037. doi:10.1214/17-BA1052. 
           https://projecteuclid.org/euclid.ba/1488337478

        """
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
    __doc__ = FastSumToZeroMultivariateNormal.__doc__

    cdef DenseMultivariateNormal mvnorm

    def __cinit__(self, random_state=None):
        self.mvnorm = DenseMultivariateNormal(random_state)

    def rvs(self, np.ndarray b, prec):
        """
        rvs(mean, prec)

        See documentation of :func:`FastSumToZeroMultivariateNormal.rvs`

        """
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
            with self.lock, nogil:
                random_int = random_positive_int64(self.rng)
        self.rng_pg = PyPolyaGamma(random_int)

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
            self.rng_pg.pgdrawv(b, z, out)
        else:
            dims = np.PyArray_DIMS(b)
            out_arr = np.PyArray_EMPTY(1, dims, np.NPY_DOUBLE, 0)
            self.rng_pg.pgdrawv(b, z, out_arr)
            return out_arr
