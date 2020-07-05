
import numpy as np
import pytest

from occuspytial.utils import get_generator, rand_precision_mat, make_data


def test_get_generator():
    rng = get_generator(0)

    assert isinstance(rng, np.random.Generator)
    # ensure the fastest SFC64 bitgenerator is used
    assert isinstance(rng.bit_generator, np.random.SFC64)

    # test if `random_state` works as expected
    rng2 = get_generator(0)

    rng2_state = rng2.bit_generator.state['state']['state']

    assert np.all(rng2_state == rng.bit_generator.state['state']['state'])


def test_random_precision_mat():
    mat = rand_precision_mat(2, 4, max_neighbors=4)
    # check the maximum number of neighbors is no more than 4
    assert mat.diagonal().max() == 3

    # check case when the max neighbors are increased
    mat = rand_precision_mat(2, 4, max_neighbors=8)
    assert mat.diagonal().max() == 5

    with pytest.raises(ValueError, match='neighbors should be one of {4, 8}'):
        rand_precision_mat(2, 4, max_neighbors=9)

    # test if output is a singular matrix (i.e rank is (n - 1))
    assert np.linalg.matrix_rank(mat.toarray()) == 7

    # test if matrix is invertible when 0 <= rh0 < 1
    mat = rand_precision_mat(2, 4, max_neighbors=8, rho=0.5)
    mat2 = rand_precision_mat(2, 4, max_neighbors=8, rho=0)
    assert np.linalg.matrix_rank(mat.toarray()) == 8
    assert np.linalg.matrix_rank(mat2.toarray()) == 8


def test_make_data():
    data = make_data(n=150, p=3, q=2, ns=65, random_state=10)
    assert data[0].shape[0] == 150
    assert data[4].shape[0] == 2
    assert data[5].shape[0] == 3
    assert data[1][3].shape[1] == 2
    assert data[2].shape[1] == 3
    assert len(data[1]) == 65

    data = make_data(n=150, p=3, q=2, random_state=10)
    assert len(data[1]) == 150 // 2

    with pytest.raises(ValueError, match='n cant be lower than'):
        make_data(n=149)

    with pytest.raises(ValueError, match='min_v needs to be at least'):
        make_data(min_v=0)

    with pytest.raises(ValueError, match='max_v is too small'):
        make_data(n=150, max_v=1)

    with pytest.raises(ValueError, match='max_v cant be more than n'):
        make_data(n=150, max_v=151)

    with pytest.raises(ValueError, match='ns should be positive'):
        make_data(ns=0)

    with pytest.raises(ValueError, match='ns cant be more than n'):
        make_data(n=150, ns=151)
