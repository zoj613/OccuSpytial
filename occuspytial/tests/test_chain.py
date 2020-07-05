import numpy as np
import pytest

from occuspytial.chain import Chain


def test_chain():
    params = {'p1':  2, 'p2': 1}
    c = Chain(params, 1)

    # test the number of columns of the full array of parameters
    assert c.full.shape[1] == 3
    assert len(c) == 0

    c.append({'p1': [1, 2], 'p2': 3})
    assert len(c) == 1

    with pytest.raises(ValueError, match='Chain is full'):
        c.append({'p1': [1, 2], 'p2': 3})

    c.expand(1)
    c.append({'p1': [1, 2], 'p2': 3})
    assert len(c) == 2

    assert np.all(c['p1'] == [[1, 2], [1, 2]])
    with pytest.raises(KeyError):
        c['p3']

    assert repr(c) == "Chain(params: ('p1', 'p2'), size: 2)"
