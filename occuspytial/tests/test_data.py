import numpy as np

from occuspytial.data import Data


def test_data():
    dic = {1: np.random.rand(5), 2: np.random.rand(4), 3: np.random.rand(2)}

    d = Data(dic)

    assert np.all(d.surveyed == [1, 2, 3])
    assert len(d) == 3
    assert np.all(d.visits([1, 3]) == (5, 2))
    assert np.all(d.visits((1, 3)) == (5, 2))
    assert d.visits(3) == 2

    assert d[1] is dic[1]

    sites = d[[1, 3]]
    sites2 = d[(1, 3)]
    assert np.allclose(sites, np.concatenate((dic[1], dic[3])))
    assert np.allclose(sites2, np.concatenate((dic[1], dic[3])))
    assert sites.size == 7
    assert sites.ndim == 1

    site = d[3]
    assert np.allclose(site, dic[3])
    assert site.size == 2
    assert site.ndim == 1

    dic = {1: np.random.rand(5, 2), 3: np.random.rand(2, 2)}

    d = Data(dic)
    assert len(d) == 2

    sites = d[[1, 3]]
    assert np.allclose(sites, np.concatenate((dic[1], dic[3])))
    assert sites.shape == (7, 2)
    assert sites.ndim == 2
