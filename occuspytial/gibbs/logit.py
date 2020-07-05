import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import solve_triangular, eigh
from scipy.sparse.linalg import eigsh

from ..distributions import (
    PolyaGamma,
    SumToZeroMultivariateNormal,
    DenseMultivariateNormal2,
)

from .base import GibbsBase
from ._cumsum import split_sum


class LogitICARGibbs(GibbsBase):

    def __init__(self, Q, W, X, y, hparams=None, random_state=None, pertub=None):
        super().__init__(Q, W, X, y, hparams, random_state)
        self._configure(Q, hparams, pertub)

    def _configure(self, Q, hparams, pertub):
        super()._configure(Q, hparams)

        if pertub is None:
            least_2_eigs = eigsh(Q, k=2, which='SA', return_eigenvectors=False)
            self.fixed.pertub = least_2_eigs[0]
        else:
            self.fixed.pertub = pertub

        random_state = self.random_state
        self.dists.pg = PolyaGamma(random_state)
        self.dists.sum2zero_mvnorm = SumToZeroMultivariateNormal(random_state)
        self.dists.mvnorm = DenseMultivariateNormal2(random_state)

    def _update_omega_a(self):
        not_obs_occupancy = [i for i in self.fixed.not_obs if self.state.z[i]]
        self.state.exists = self.fixed.obs + not_obs_occupancy
        self.state.W = self.W[self.state.exists]
        b = self.state.W @ self.state.alpha
        self.dists.pg.rvs(np.ones_like(b), b, b)
        self.state.omega_a = b

    def _update_omega_b(self):
        b = self.X @ self.state.beta + self.state.spatial
        self.dists.pg.rvs(self.fixed.ones, b, b)
        self.state.omega_b = b

    def _update_tau(self):
        eta = self.state.eta
        rate = 0.5 * eta @ self.fixed.Q @ eta + self.fixed.tau_rate
        self.state.tau = self.rng.gamma(self.fixed.tau_shape, 1 / rate)

    def _update_eta(self):
        A = self.fixed.Q.copy()
        A.data = A.data * self.state.tau
        A.setdiag(A.diagonal() + self.state.omega_b + self.fixed.pertub)
        b = self.state.k - (self.state.omega_b * (self.X @ self.state.beta))
        self.state.eta = self.dists.sum2zero_mvnorm.rvs(b, A)
        self.state.spatial = self.state.eta

    def _update_alpha(self):
        W = self.state.W
        y = self.y[self.state.exists] - 0.5
        A = (W.T * self.state.omega_a) @ W + self.fixed.a_prec
        b = W.T @ y + self.fixed.a_prec_by_mu
        self.state.alpha = self.dists.mvnorm.rvs(b, A)

    def _update_beta(self):
        spat = self.state.spatial
        omega = self.state.omega_b
        A = (self.X.T * omega) @ self.X + self.fixed.b_prec
        b = self.X.T @ (self.state.k - (omega * spat)) + self.fixed.b_prec_by_mu
        self.state.beta = self.dists.mvnorm.rvs(b, A)

    def _update_z(self):
        no = self.fixed.not_obs
        n_no = self.fixed.n_no
        ns = self.fixed.not_surveyed
        n_ns = self.fixed.n_ns
        beta = self.state.beta
        spat = self.state.spatial

        xb_eta = self.X[no] @ beta + spat[no]
        y = -np.logaddexp(0, -xb_eta)
        w_a = self.fixed.W_not_obs @ self.state.alpha
        omd = -np.logaddexp(0, w_a)
        split_sum(omd, self.fixed.sections, self.state.section_sums)
        x = y + self.state.section_sums
        c = -np.logaddexp(0, xb_eta)
        logp = x - np.logaddexp(c, x)
        self.state.z[no] = np.log(self.rng.uniform(size=n_no)) < logp

        if ns:
            xb_eta_ns = self.X[ns] @ beta + spat[ns]
            logp = -np.logaddexp(0, -xb_eta_ns)
            self.state.z[ns] = np.log(self.rng.uniform(size=n_ns)) < logp

        self.state.k = self.state.z - 0.5

    def step(self):
        self._update_omega_b()
        self._update_tau()
        self._update_eta()
        self._update_beta()
        self._update_omega_a()
        self._update_alpha()
        self._update_z()


class LogitRSRGibbs(LogitICARGibbs):
    def __init__(
        self, Q, W, X, y, hparams=None, random_state=None, r=0.5, q=None,
    ):
        super().__init__(Q, W, X, y, hparams, random_state)
        self._configure_rsr(r, q, hparams)

    def _configure_rsr(self, r, q, hparams):
        # XTX_i = inv(self.X.T @ self.X)
        chol = np.linalg.cholesky(self.X.T @ self.X)
        z = solve_triangular(chol, np.eye(self.X.shape[1]), lower=True)
        XTX_i = solve_triangular(chol, z, lower=True, trans=1)

        # P = I - X @ XTX_i @ XT
        P = -multi_dot([self.X, XTX_i, self.X.T])
        P[np.diag_indices_from(P)] += 1

        A = self.fixed.Q.copy()
        A.data = -A.data
        A.setdiag(0)
        omega = self.fixed.n * (P.T @ A @ P) / A.sum()
        # return eigen vectors of omega and keep first q columns
        # corresponding to the q largest eigenvalues of omega greater
        # than the specified threshold
        w, v = eigh(omega, overwrite_a=True)
        # order eigens in descending order
        w, v = w[::-1], np.fliplr(v)
        if q:
            self.fixed.q = q
        else:
            if not 0 <= r <= 1:
                raise ValueError('Threshold value needs to be in [0, 1]')
            # number of eigenvalues > threshold
            self.fixed.q = w[w >= r].size
            if not self.fixed.q:
                raise ValueError(
                    'The Moran Operator Matrix of the data has no positive '
                    'eigenvalues. Set threshold to a lower value'
                )
        # keep first q eigenvectors of ordered eigens
        K = v[:, :self.fixed.q]
        # replace Q with Minv
        Q_copy = self.fixed.Q
        del self.fixed.Q
        self.fixed.Q = K.T @ Q_copy @ K
        self.fixed.K = K
        if not hparams:
            # `_set_default_hyperparams` has been called so modify tau_shape
            del self.fixed.tau_shape
            self.fixed.tau_shape = 0.5 + 0.5 * self.fixed.q

    def _initialize_default_start(self, state):
        state = super()._initialize_default_start(state)
        state.eta = self.rng.normal(scale=5, size=self.fixed.q)
        state.spatial = self.fixed.K @ self.state.eta
        return state

    def _initialize_posterior_state(self, start=None):
        if start is None:
            self.state = self._initialize_default_start(self.state)
        else:
            self.state.alpha = start['alpha']
            self.state.beta = start['beta']
            self.state.tau = start['tau']
            self.state.eta = start['eta']
            self.state.spatial = self.fixed.K @ self.state.eta

    def _update_eta(self):
        K = self.fixed.K
        omega = self.state.omega_b
        A = (K.T * omega) @ K + (self.state.tau * self.fixed.Q)
        b = K.T @ (self.state.k - omega * (self.X @ self.state.beta))
        self.state.eta = self.dists.mvnorm.rvs(b, A)
        self.state.spatial = self.fixed.K @ self.state.eta
