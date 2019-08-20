import pytest
import numpy as np
from scipy.sparse import csc_matrix

from occuspytial.utils.dataprocessing import SpatialStructure
from occuspytial.utils.misc import CustomDict
from occuspytial.utils.stats import acf, affine_sample


@pytest.fixture
def x():
    """returns a random series of 20 values"""
    return np.random.rand(20)


def test_acf_raises_exception_on_incorrect_lag(x):
    lag = len(x)
    with pytest.raises(Exception):
        acf(x, lag)


def test_acf_zero_lag(x):
    assert acf(x, 0) == 1


def test_acf_works_with_negative_lag(x):
    assert acf(x, -5) == acf(x, 5)


@pytest.fixture
def dic():
    out = dict(
        a=np.random.rand(2, 3),
        b=np.random.rand(4, 3),
        c=np.random.rand(1, 3)
    )
    return CustomDict(out)


def test_customdict_hashable_indexing(dic):
    expected = np.concatenate((dic["a"], dic["c"]))
    np.testing.assert_equal(dic.slice("a", "c"), expected)


@pytest.mark.parametrize(
    "indices",
    [np.array(["a", "c"]), ["a", "c"]]
)
def test_customdict_nonhashable_indexing(dic, indices):
    expected = np.concatenate((dic["a"], dic["c"]))
    np.testing.assert_equal(dic.slice(indices), expected)


def test_icar_spatial_precision_is_singular():
    mat = SpatialStructure(100).spatial_precision()
    assert np.linalg.matrix_rank(mat) < mat.shape[0]


def test_car_spatial_precision_is_invertible():
    mat = SpatialStructure(100).spatial_precision(rho=0.5)
    assert np.linalg.matrix_rank(mat) == mat.shape[0]


@pytest.fixture
def mean_cov():
    mean, cov = np.random.rand(50), np.random.rand(50, 50)
    return mean, cov.T @ cov


@pytest.fixture
def sp_mean_cov(mean_cov):
    """get sparse covariance"""
    mean, cov = mean_cov
    return mean, csc_matrix(cov)


def test_affine_sample_works_for_dense_covarince_input(mean_cov):
    _, factor = affine_sample(*mean_cov, return_factor=True)
    assert isinstance(factor, np.ndarray)


def test_affine_sample_works_for_sparse_covarince_input(sp_mean_cov):
    try:
        _, factor = affine_sample(*sp_mean_cov, return_factor=True)
        from sksparse.cholmod import Factor
        assert isinstance(factor, Factor)
    except ImportError:
        # this is for when scikit-sparse isnt installed, which should raise
        # an ImportError since issparse() returns false for a numpy array
        assert isinstance(factor, np.ndarray)


def test_affine_sample_factor_reconstructs_covariance(sp_mean_cov):
    try:
        mean, sp_cov = sp_mean_cov
        _, factor = affine_sample(mean, sp_cov, return_factor=True)
        chol = factor.apply_Pt(factor.L()).toarray()
        np.testing.assert_array_almost_equal(chol @ chol.T, sp_cov.toarray())
    except AttributeError:
        # this is for when scikit-sparse isnt installed, which willl raise
        # an AttributeErro since a numpy array doesnt have the apply_Pt()
        # attribute.
        chol = factor
        np.testing.assert_array_almost_equal(chol @ chol.T, sp_cov.toarray())
