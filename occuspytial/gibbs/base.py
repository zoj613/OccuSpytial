from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from ..chain import Chain
from ..data import Data
from ..posterior import PosteriorParameter

from .parallel import sample_parallel
from .state import State, FixedState


class _GibbsState(State):
    _posterior_names = ('alpha', 'beta', 'tau', 'z', 'eta')

    @property
    def posteriors(self):
        return {key: self.__dict__[key] for key in self._posterior_names}


class GibbsBase(ABC):
    def __init__(self, Q, W, X, y, hparams=None, random_state=None):
        self.W = Data(W)
        self.X = X
        self.y = Data(y)
        self.chain = None
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    @abstractmethod
    def step(self):
        pass

    def _configure(self, Q, hparams, verify_precision=True, **kwargs):
        if verify_precision:
            self._verify_spatial_precision(Q)

        self.state = _GibbsState()
        self.state.z = np.ones(self.X.shape[0])
        self.state.z[self.y.surveyed] = [
            0 if not any(self.y[site]) else 1 for site in self.y.surveyed
        ]
        self.state.k = self.state.z - 0.5

        self.fixed = FixedState()
        self.fixed.n = self.X.shape[0]
        self.fixed.ones = np.ones(self.fixed.n)
        self.fixed.zeros = np.zeros(self.fixed.n)
        self.fixed.not_surveyed = [
            site for site in range(self.fixed.n) if site not in self.y.surveyed
        ]
        self.fixed.not_obs = [i for i in self.y.surveyed if not self.state.z[i]]
        self.fixed.obs = [i for i in self.y.surveyed if self.state.z[i]]
        self.fixed.W_not_obs = self.W[self.fixed.not_obs]
        self.fixed.visits_not_obs = self.W.visits(self.fixed.not_obs)
        sections = np.cumsum(self.fixed.visits_not_obs)
        self.fixed.sections = np.pad(sections, (1, 0))
        self.state.section_sums = np.zeros(sections.shape[0], dtype=np.double)
        self.fixed.Q = Q if isspmatrix_csc(Q) else csc_matrix(Q)

        if hparams:
            self.fixed = self._set_hyperparams(self.fixed, hparams)
        else:
            self.fixed = self._set_default_hyperparams(self.fixed)
        self.fixed.a_prec_by_mu = self.fixed.a_prec @ self.fixed.a_mu
        self.fixed.b_prec_by_mu = self.fixed.b_prec @ self.fixed.b_mu

        self.dists = FixedState()

    def _verify_spatial_precision(self, Q):
        smallest_eigv = eigsh(Q, k=1, which='SA', return_eigenvectors=False)[0]
        if smallest_eigv >= 1e-4:
            raise ValueError('Spatial precision matrix Q must be singular.')

    def _set_hyperparams(self, params, hyperparams):
        for key, value in hyperparams.items():
            params[key] = value
        return params

    def _set_default_hyperparams(self, params):
        params.tau_rate = 0.005
        params.tau_shape = 0.5 + 0.5 * (self.fixed.n - 1)
        alpha_size = self.W[self.W.surveyed[0]].shape[1]
        params.a_mu = np.zeros(alpha_size)
        params.a_prec = np.eye(alpha_size) / 10
        beta_size = self.X.shape[1]
        params.b_mu = np.zeros(beta_size)
        params.b_prec = np.eye(beta_size) / 10
        return params

    def _initialize_posterior_state(self, start=None):
        if start is None:
            self.state = self._initialize_default_start(self.state)
        else:
            self.state.alpha = start['alpha']
            self.state.beta = start['beta']
            self.state.tau = start['tau']
            self.state.eta = start['eta']
            self.state.spatial = self.state.eta

    def _initialize_default_start(self, state):
        state.tau = self.rng.gamma(0.5, 1 / self.fixed.tau_rate)
        eta = self.rng.standard_normal(self.fixed.n)
        eta = eta - eta.mean()
        state.eta = eta
        state.spatial = self.state.eta
        state.alpha = self.rng.multivariate_normal(
            self.fixed.a_mu, 100 * self.fixed.a_prec
        )
        state.beta = self.rng.multivariate_normal(
            self.fixed.b_mu, 100 * self.fixed.b_prec
        )
        return state

    def _run(
        self, size, burnin=0, start=None, chains=1, progressbar=True, pos=0
    ):
        self._initialize_posterior_state(start)
        chain_params = {
            'alpha': self.state.alpha.size,
            'beta':  self.state.beta.size,
            'eta':  self.state.eta.size,
            'z':  self.fixed.n,
            'tau': 1
        }
        self.chain = Chain(chain_params, size - burnin)
        tqdm_iterator = tqdm(
            range(size), total=size, disable=not progressbar, position=pos
        )
        for i in tqdm_iterator:
            self.step()
            if i >= burnin:
                self.chain.append(self.state.posteriors)

        return self.chain

    def sample(
        self, size, burnin=0, start=None, chains=1, progressbar=True, pos=0
    ):
        if burnin >= size:
            raise ValueError('burnin value cannot be larger than sample size')
        if chains < 1:
            raise ValueError('chains must a postive integer.')

        samples = sample_parallel(
            self, size=size, burnin=burnin, chains=chains, start=start
        )
        out = PosteriorParameter(*samples)
        return out

    def resample(self, chain, size):
        # TODO: clean this up and improve efficiency
        if not isinstance(chain, PosteriorParameter):
            raise ValueError(
                '"chain" needs to be an instance of PosteriorParameter'
            )
        chain = deepcopy(chain)
        out = chain.data.pad(
            pad_width={'draw': (0, size)}, mode='constant', constant_values=0
        )
        out['draw'] = np.arange(out.draw.size)
        # pad the exluded params
        for key in chain._excluded:
            chain._excluded[key] = np.pad(
                chain._excluded[key], [(0, 0), (0, size), (0, 0)]
            )
        for i in range(len(chain.data.chain)):
            start = {
                key: val[i][-(size + 1)]
                for key, val in chain._excluded.items()
            }
            start.update({
                k: chain[k][i][-1] for k in chain.data.keys()
            })
            new_chain = self.sample(size, burnin=0, start=start)
            for key in chain.data.keys():
                out[key].data[i][-size:] = new_chain[key]
            for key, val in chain._excluded.items():
                chain._excluded[key][i][-size:] = new_chain._excluded[key][0]
        chain.data = out
        return chain

    def copy(self):
        out = type(self).__new__(self.__class__)
        out.__dict__.update(self.__dict__)
        # make sure the copy has its own unique random number generator
        out.__dict__['rng'] = np.random.default_rng()
        return out
