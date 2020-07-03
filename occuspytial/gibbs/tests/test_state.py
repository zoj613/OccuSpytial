import pytest

from occuspytial.gibbs.state import State, FixedState


def test_state():
    s = State()

    s.a = 1
    s.b = 2

    with pytest.raises(TypeError, match='does not support item assignment'):
        s['b'] = 2

    assert s.a == 1
    s.a = 0.5
    assert s.a == 0.5

    items = [i for i in s]
    assert len(items) == 2
    assert items[0] == 'a'
    assert items[1] == 'b'

    fs = FixedState()

    fs.c = 3
    with pytest.raises(KeyError, match='cannot change attributes already set'):
        fs.c = 2
