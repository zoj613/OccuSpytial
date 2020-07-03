import numpy as np
import pytest
from scipy.sparse import diags
try:
    from sksparse.cholmod import Factor
    HAS_SKSPARSE = True
except ImportError:
    HAS_SKSPARSE = False

from occuspytial.distributions import (
    FastSumToZeroMultivariateNormal,
    SlowSumToZeroMultivariateNormal,
    SparseMultivariateNormal,
    DenseMultivariateNormal,
    PolyaGamma,
)

skip_if_no_skparse = pytest.mark.skipif(
    not HAS_SKSPARSE, reason='scikit-sparse is not installed'
)


@skip_if_no_skparse
def test_sparse_mvnorm():
    d = SparseMultivariateNormal()
    cov = diags(np.random.rand(10), format='csc')
    mean = np.array([0.] * 10)

    # check if sampling is successful
    arr, factor = d.rvs(mean, cov)
    assert isinstance(arr, np.ndarray)
    assert isinstance(factor, Factor)
    assert arr.ndim == 1
    assert arr.shape[0] == 10
    # check failure on non-sparse covariance
    with pytest.raises(AttributeError, match="has no attribute 'tocsc'"):
        d.rvs(mean, cov.toarray())

    with pytest.raises(ValueError, match='dimension mismatch'):
        d.rvs(np.array([0.] * 11), cov)


def test_dense_mvnorm():
    d = DenseMultivariateNormal()
    mat = np.random.rand(5, 5)
    cov = np.asfortranarray(mat.T @ mat)
    mean = np.array([0.] * 5)

    with pytest.raises(ValueError, match='ndarray is not Fortran contiguous'):
        d.rvs(mean, np.ascontiguousarray(cov))

    cov_copy = cov.copy(order='F')
    arr, factor = d.rvs(mean, cov_copy)
    factor_arr = np.asarray(factor)
    factor_arr[np.tril_indices_from(factor_arr, k=-1)] = 0

    assert isinstance(arr, np.ndarray)
    # check if input covariance was overwritten
    assert np.allclose(factor, cov_copy)
    # check if input factor can reconstruct the input
    assert np.allclose(factor_arr.T @ factor_arr, cov)
    assert arr.ndim == 1
    assert arr.shape[0] == 5

    arr, factor = d.rvs(mean, cov, overwrite_cov=False)
    assert not np.allclose(factor, cov)

    with pytest.raises(RuntimeError):
        d.rvs(np.array([0.] * 6), cov)


@skip_if_no_skparse
def test_fast_sum_to_zero_mvnorm():
    d = FastSumToZeroMultivariateNormal()
    prec = diags(np.random.rand(10), format='csc')
    b = np.random.rand(10)

    arr = d.rvs(b, prec)
    assert arr.sum() < 1e-10


def test_slow_sum_to_zero_mvnorm():
    d = SlowSumToZeroMultivariateNormal()
    prec = diags(np.random.rand(10), format='csc')
    b = np.random.rand(10)

    arr = d.rvs(b, prec)
    assert arr.sum() < 1e-10


def test_polyagamma():
    d = PolyaGamma()

    out = d.rvs(1, 1)
    assert isinstance(out, float)
    assert out > 0

    a = np.array([1.] * 5)
    b = np.array([2.] * 5)

    out = d.rvs(a, b)
    assert out.size == a.size == b.size

    c = b.copy()
    d.rvs(a, b, out=b)
    assert not np.allclose(c, b)

    with pytest.raises(ValueError, match="expected 'double' but got 'long'"):
        d.rvs(np.array([0] * 5), b)
