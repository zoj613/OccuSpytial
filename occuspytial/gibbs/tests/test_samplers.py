import numpy as np
import pytest
from scipy.sparse import rand

from occuspytial.gibbs.base import GibbsBase
from occuspytial.gibbs.logit import LogitICARGibbs, LogitRSRGibbs
from occuspytial.gibbs.probit import ProbitRSRGibbs
from occuspytial.utils import make_data, get_generator


rng = get_generator(random_state=10)

# TODO maybe turn this into a fixture?
Q, W, X, y, alpha, beta, tau, z = make_data(
    min_v=2, max_v=10, ns=100, p=3, q=2, tau_range=(0.25, 1.5), random_state=10
)

hypers = {
    'tau_rate': 1.0,
    'tau_shape': 5.0,
    'a_mu': rng.random(2),
    'b_mu': rng.random(3),
    'a_prec': np.eye(2),
    'b_prec': np.eye(3)
}


parametrized_samplers = pytest.mark.parametrize(
    'sampler_class',
    [
        LogitRSRGibbs,
        LogitICARGibbs,
        pytest.param(ProbitRSRGibbs, marks=pytest.mark.xfail)
    ]
)


def test_progressbar_output(capfd):
    s = LogitICARGibbs(Q, W, X, y)
    s.sample(10)
    captured = capfd.readouterr()
    # test if progressbar was displayed
    assert ' 10/10 [00:00<00:00,' in captured.err


def test_turning_off_progressbar(capfd):
    s = LogitICARGibbs(Q, W, X, y)
    s.sample(10, progressbar=False)
    captured = capfd.readouterr()
    # test if progressbar was not displayed
    assert ' 10/10 [00:00<00:00,' not in captured.err


@parametrized_samplers
def test_gibbs_samplers(sampler_class):
    s = sampler_class(Q, W, X, y, random_state=10)
    samples = s.sample(5)
    # test if correct number of samples is generated per parameter
    assert samples['alpha'].shape == (1, 5, 2)
    assert samples['beta'].shape == (1, 5, 3)
    assert samples['tau'].shape == (1, 5)

    # test reproducability through random_state
    s = sampler_class(Q, W, X, y, random_state=10)
    samples2 = s.sample(5)
    assert np.allclose(samples2['alpha'], samples['alpha'])
    assert np.allclose(samples2['beta'], samples['beta'])
    assert np.allclose(samples2['tau'], samples['tau'])

    # test copy method
    s_copy = s.copy()
    assert isinstance(s_copy, sampler_class)

    # test if burnin works as intended
    with pytest.raises(ValueError, match='burnin value cannot be larger than'):
        s.sample(10, burnin=11)
    samples = s.sample(10, burnin=3)
    assert samples['alpha'].shape == (1, 7, 2)
    assert samples['beta'].shape == (1, 7, 3)
    assert samples['tau'].shape == (1, 7)

    with pytest.raises(ValueError, match='chains must a positive integer'):
        s.sample(10, chains=0)
    samples = s.sample(5, chains=3)
    assert samples['alpha'].shape == (3, 5, 2)
    assert samples['beta'].shape == (3, 5, 3)
    assert samples['tau'].shape == (3, 5)


@pytest.mark.parametrize(
    'sampler, start',
    [
        (
            LogitICARGibbs(Q, W, X, y, random_state=10),
            {'eta': rng.random(150)}
        ),
        (
            LogitRSRGibbs(Q, W, X, y, random_state=10, q=10),
            {'eta': rng.random(10)}
        ),
        pytest.param(
            ProbitRSRGibbs(Q, W, X, y, random_state=10, q=10),
            {'eta': rng.random(10), 'eps': rng.standard_normal(150)},
            marks=pytest.mark.xfail
        )
    ]
)
def test_sampler_start_parameter(sampler, start):
    s = sampler
    samples = s.sample(5)
    # test `start` parameter works as intended
    s = sampler
    _start = {'alpha': rng.random(2), 'beta': rng.random(3), 'tau': 2}
    _start.update(start)
    samples2 = s.sample(5, start=_start)
    assert not np.allclose(samples2['alpha'][0, 0], samples['alpha'][0, 0])
    assert not np.allclose(samples2['beta'][0, 0], samples['beta'][0, 0])
    assert not np.allclose(samples2['tau'][0, 0], samples['tau'][0, 0])


@pytest.mark.parametrize(
    'sampler_class, params',
    [
        (LogitRSRGibbs, {'Q': Q, 'W': W, 'X': X, 'y': y, 'r': 1.1}),
        pytest.param(
            ProbitRSRGibbs,
            {'Q': Q, 'W': W, 'X': X, 'y': y, 'r': 1.1},
            marks=pytest.mark.xfail
        )
    ]
)
def test_rsr_sampler_threshold_parameter(sampler_class, params):
    with pytest.raises(ValueError, match='Threshold value needs to be in'):
        sampler_class(**params)


@parametrized_samplers
def test_hyperameter_input(sampler_class):
    s1 = sampler_class(Q, W, X, y)
    s2 = sampler_class(Q, W, X, y, hparams=hypers)

    assert s1.fixed['tau_shape'] != s2.fixed['tau_shape']
    assert s1.fixed.tau_rate != s2.fixed.tau_rate
    assert not np.allclose(s1.fixed.a_mu, s2.fixed.a_mu)
    assert not np.allclose(s1.fixed.b_mu, s2.fixed.b_mu)
    assert not np.allclose(s1.fixed.a_prec, s2.fixed.a_prec)
    assert not np.allclose(s1.fixed.b_prec, s2.fixed.b_prec)


@pytest.fixture
def nonsingular_mat():
    mat = rand(150, 150, density=0.9, format='csc', random_state=10)
    return mat.T * mat


@parametrized_samplers
def test_nonsingular_spatial_precision_matrix(sampler_class, nonsingular_mat):
    with pytest.raises(ValueError, match='Spatial precision matrix Q must be'):
        sampler_class(nonsingular_mat, W, X, y)


def test_sampler_with_no_step_method():
    class FakeSampler(GibbsBase):
        def __init__(self, Q, W, X, y):
            super().__init__(Q, W, X, y)
            super()._configure(Q, None)

    msg = 'FakeSampler must implement a `step` method.'
    with pytest.raises(NotImplementedError, match=msg):
        s = FakeSampler(Q, W, X, y)
        s.sample(5)
