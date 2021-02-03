import numpy as np

from occuspytial.distributions import precision_mvnorm


def test_precision_mvnorm():
    mat = np.random.rand(5, 5)
    prec_mat = np.linalg.inv(mat.T @ mat)
    mean = np.array([0.] * 5)

    prec_mat_copy = prec_mat.copy()
    arr = precision_mvnorm(mean, prec_mat_copy)

    assert arr.ndim == 1
    assert arr.shape[0] == 5
    # check if input precision matrix was overwritten
    assert not np.allclose(prec_mat, prec_mat_copy)
    # test reproducibility
    prec_mat_copy = prec_mat.copy()
    rng = np.random.default_rng(0)
    arr1 = precision_mvnorm(mean, prec_mat_copy, random_state=rng)
    rng = np.random.default_rng(0)
    prec_mat_copy = prec_mat.copy()
    arr2 = precision_mvnorm(mean, prec_mat_copy, random_state=rng)
    assert np.allclose(arr1, arr2)
    assert not np.allclose(arr1, arr)
