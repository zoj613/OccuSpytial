import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import solve_triangular, eigh
from scipy.special import ndtr, log_ndtr  # std norm cdf and its log
from scipy.stats import truncnorm, norm

from ..distributions import DenseMultivariateNormal2

from .base import GibbsBase
from ._cumsum import split_sum


class ProbitRSRGibbs(GibbsBase):
    def __init__(
        self, Q, W, X, y, hparams=None, random_state=None, r=0.5, q=None
    ):
        super().__init__(Q, W, X, y, hparams, random_state)
        self._configure(Q, hparams, q, r)

    def _configure(self, Q, hparams, q, r):
        super()._configure(Q, hparams)
        self.state.omega_b = np.zeros(self.fixed.n)
        self.fixed.XTX_plus_bprec = self.X.T @ self.X + self.fixed.b_prec
        self.fixed.eps_chol_factor = np.ones(self.X.shape[0]) / np.sqrt(2)
        self.dists.mvnorm = DenseMultivariateNormal2()

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
        self.fixed.KTK = K.T @ K
        # replace Q with Minv
        Q_copy = self.fixed.Q
        del self.fixed.Q
        self.fixed.Q = K.T @ Q_copy @ K
        self.fixed.K = K
        # `_set_default_hyperparams` has been called so modify tau_shape
        del self.fixed.tau_shape
        self.fixed.tau_shape = 0.5 + 0.5 * self.fixed.q

    def _initialize_default_start(self, state):
        state = super()._initialize_default_start(state)
        state.eta = self.rng.normal(scale=5, size=self.fixed.q)
        state.spatial = self.fixed.K @ self.state.eta
        state.eps = self.rng.standard_normal(self.fixed.n)
        return state

    def _initialize_posterior_state(self, start=None):
        if start is None:
            self.state = self._initialize_default_start(self.state)
        else:
            self.state.alpha = start['alpha']
            self.state.beta = start['beta']
            self.state.tau = start['tau']
            self.state.eta = start['eta']
            self.state.eps = start['eps']
            self.state.spatial = self.fixed.K @ self.state.eta

    def _update_omega_a(self):
        self.state.exists = self.state.z[self.W.surveyed].nonzero()[0]
        obs_mask = (self.y[self.state.exists] == 1)
        self.state.W = self.W[self.state.exists]
        loc = self.state.W @ self.state.alpha
        self.state.omega_a = np.zeros_like(loc)
        a = loc[obs_mask]
        self.state.omega_a[obs_mask] = truncnorm(-a, np.inf, loc=a).rvs()
        b = loc[~obs_mask]
        self.state.omega_a[~obs_mask] = truncnorm(-np.inf, -b, loc=b).rvs()

    def _update_omega_b(self):
        loc = self.X @ self.state.beta + self.state.spatial + self.state.eps
        exist_mask = (self.state.z == 1)
        a = loc[exist_mask]
        self.state.omega_b[exist_mask] = truncnorm(-a, np.inf, loc=a).rvs()
        b = loc[~exist_mask]
        self.state.omega_b[~exist_mask] = truncnorm(-np.inf, -b, loc=b).rvs()

    def _update_tau(self):
        eta = self.state.eta
        rate = (0.5 * eta @ self.fixed.Q @ eta) + self.fixed.tau_rate
        self.state.tau = self.rng.gamma(self.fixed.tau_shape, 1 / rate)

    def _update_eps(self):
        mean = 0.5 * (
            self.state.omega_b - self.X @ self.state.beta - self.state.spatial
        )
        std = self.rng.standard_normal(mean.size)
        self.state.eps = mean + self.fixed.eps_chol_factor * std

    def _update_eta(self):
        K = self.fixed.K
        eps = self.state.eps
        A = self.fixed.KTK + self.state.tau * self.fixed.Q
        b = K.T @ (self.state.omega_b - self.X @ self.state.beta - eps)
        self.state.eta = self.dists.mvnorm.rvs(b, A)
        self.state.spatial = self.fixed.K @ self.state.eta

    def _update_alpha(self):
        W = self.state.W
        A = W.T @ W + self.fixed.a_prec
        b = self.fixed.a_prec_by_mu + W.T @ self.state.omega_a
        self.state.alpha = self.dists.mvnorm.rvs(b, A)

    def _update_beta(self):
        b = self.fixed.b_prec_by_mu + self.X.T @ (
            self.state.omega_b - self.state.spatial - self.state.eps
        )
        self.state.beta = self.dists.mvnorm.rvs(b, self.fixed.XTX_plus_bprec)

    def _update_z(self):
        no = self.fixed.not_obs
        ns = self.fixed.not_surveyed
        beta = self.state.beta
        K_eta = self.state.spatial
        xb_eta = self.X[no] @ beta + K_eta[no] + self.state.eps[no]
        w_a = self.fixed.W_not_obs @ self.state.alpha
        num1 = ndtr(xb_eta)
        lognum1 = log_ndtr(xb_eta)
        lognum2 = norm.logsf(w_a)
        split_sum(lognum2, self.fixed.sections, self.state.section_sums)
        lognum = lognum1 + self.state.section_sums
        prod_sf = np.exp(self.state.section_sums)
        logp = lognum - np.log(1 - num1 + num1 * prod_sf)
        self.state.z[no] = self.rng.binomial(n=1, p=np.exp(logp))

        if ns:
            p = ndtr(self.X[ns] @ beta + K_eta[ns] + self.state.eps[ns])
            self.state.z[ns] = self.rng.binomial(n=1, p=p)

    def step(self):
        self._update_omega_b()
        self._update_tau()
        self._update_eps()
        self._update_eta()
        self._update_beta()
        self._update_omega_a()
        self._update_alpha()
        self._update_z()