import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import solve_triangular, eigh
from scipy.special import ndtr, log_ndtr  # std norm cdf and its log
from scipy.stats import truncnorm

from ..distributions import DenseMultivariateNormal2

from .base import GibbsBase


class ProbitRSRGibbs(GibbsBase):

    """Sampler using probit link and RSR model for spatial random effects.

    This algorithm is an implementation of the gibbs sampler in [1]_ where a
    Reduced Spatial Regression (RSR) model is used to account for spatial
    autocorrelation in a single-season site occupancy model, but a probit
    link function is used instead of a logit.

    Parameters
    ----------
    Q : np.ndarray
        Spatial precision matrix of spatial random effects.
    W : Dict[int, np.ndarray]
        Dictionary of detection corariates where the keys are the site numbers
        of the surveyed sites and the values are arrays containing
        the design matrix of each corresponding site.
    X : np.ndarray
        Design matrix of species occupancy covariates.
    y : Dict[int, np.ndarray]
        Dictionary of survey data where the keys are the site numbers of the
        surveyed sites and the values are number arrays of 1's and 0's
        where 0's indicate "no detection" and 1's indicate "detection". The
        length of each array equals the number of visits in the corresponding
        site.
    hparams : {None, Dict[str, Union[float, np.ndarray]}, optional
        Hyperparameters of the occupancy model. valid keys for the dictionary
        are:
            - ``alpha`` : coefficients of conditional detection covariates.
            - ``beta`` : coefficients of occupancy covariates
            - ``tau`` : spatial precision parameter
    random_state : {None, int, numpy.random.SeedSequence}
        A seed to initialize the bitgenerator.
    r : float, optional
        The threshold of non-negative eigenvalues to keep of the Moran matrix
        to form the RSR precision matrix. Defaults to 0.5, meaning only
        columns of the Moran matrix that have corresponding eigenvalues
        greater than 0.5 will be used. If `q` is set, then this parameter is
        ignored.
    q : int, optional
        The number of columns of the Moran matrix to use in order to form the
        spatial precision matrix of the RSR model. If this parameter is used,
        then the value of the `r` parameter is ignore. If the value is None,
        then the default value of `r` is used to create the spatial precision
        matrix of the RSR model. Defaults to None.

    Methods
    -------
    sample(size, burnin=0, start=None, chains=1, progressbar=True)

    See Also
    --------
    occuspytial.gibbs.probit.LogitRSRGibbs :
        The RSR gibbs sampler using a Logit link function.

    Notes
    -----
    When the assumption that the random effect :math:`\\epsilon` is normally
    distributed fails to hold, then a functional form misspecification issue
    arises: if the model is still estimated as a probit model, the estimators
    of the coefficients :math:`\\beta` are inconsistent [1]_.

    References
    ----------
    .. [1]  Joachim Inkmann(2000). Misspecified heteroskedasticity in the panel
       probit model: A small sample comparison of GMM and SML estimators.
       Journal of Econometrics, 97(2), 227-259

    """
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
        random_state = self.rng.integers(low=0, high=2 ** 63)
        self.dists.mvnorm = DenseMultivariateNormal2(random_state)

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

        if not hparams:
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
        """Update the latent variable associated with the cofficients of the
        conditional detection covariates.
        """
        # get occopancy state of the sites where species was not observed.
        not_obs_occupancy = [i for i in self.fixed.not_obs if self.state.z[i]]
        self.state.exists = self.fixed.obs + not_obs_occupancy
        obs_mask = (self.y[self.state.exists] == 1)
        self.state.W = self.W[self.state.exists]
        loc = self.state.W @ self.state.alpha
        random_state = self.rng.integers(low=0, high=2 ** 32 - 1)
        self.state.omega_a = np.zeros_like(loc)
        a = loc[obs_mask]
        self.state.omega_a[obs_mask] = truncnorm(-a, np.inf, loc=a).rvs(
            random_state=random_state
        )
        b = loc[~obs_mask]
        self.state.omega_a[~obs_mask] = truncnorm(-np.inf, -b, loc=b).rvs(
            random_state=random_state
        )

    def _update_omega_b(self):
        """Update the latent variable associated with the cofficients of the
        occupancy covariates.
        """
        loc = self.X @ self.state.beta + self.state.spatial + self.state.eps
        exist_mask = (self.state.z == 1)
        random_state = self.rng.integers(low=0, high=2 ** 32 - 1)
        a = loc[exist_mask]
        self.state.omega_b[exist_mask] = truncnorm(-a, np.inf, loc=a).rvs(
            random_state=random_state
        )
        b = loc[~exist_mask]
        self.state.omega_b[~exist_mask] = truncnorm(-np.inf, -b, loc=b).rvs(
            random_state=random_state
        )

    def _update_tau(self):
        eta = self.state.eta
        rate = 0.5 * eta @ self.fixed.Q @ eta + self.fixed.tau_rate
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
        WT = self.state.W.T
        A = WT @ WT.T + self.fixed.a_prec
        b = self.fixed.a_prec_by_mu + WT @ self.state.omega_a
        self.state.alpha = self.dists.mvnorm.rvs(b, A)

    def _update_beta(self):
        b = self.fixed.b_prec_by_mu + self.X.T @ (
            self.state.omega_b - self.state.spatial - self.state.eps
        )
        self.state.beta = self.dists.mvnorm.rvs(b, self.fixed.XTX_plus_bprec)

    def _update_z(self):
        no = self.fixed.not_obs
        n_no = self.fixed.n_no
        ns = self.fixed.not_surveyed
        n_ns = self.fixed.n_ns
        beta = self.state.beta
        K_eta = self.state.spatial
        xb_eta = self.X[no] @ beta + K_eta[no] + self.state.eps[no]
        w_a = self.fixed.W_not_obs @ self.state.alpha
        num1 = ndtr(xb_eta)
        lognum1 = log_ndtr(xb_eta)
        # clip the values of the survival function to be no less than 1e-4
        # as an attempt to stabilize taking the log of a nearly zero value
        # and prevent the output from being a -infinity which causes the
        # binomial probabilities to go outside the range [0, 1] for those
        # values.
        num2sf = np.clip(1 - ndtr(w_a), a_min=1e-4, a_max=np.inf)
        lognum2 = np.log(num2sf)
        stack_sum = np.add.reduceat(lognum2, self.fixed.stacked_w_indices)
        lognum = lognum1 + stack_sum
        prod_sf = np.exp(stack_sum)
        logp = lognum - np.log(1 - num1 + num1 * prod_sf)
        self.state.z[no] = np.log(self.rng.uniform(size=n_no)) < logp

        if ns:
            p = ndtr(self.X[ns] @ beta + K_eta[ns] + self.state.eps[ns])
            self.state.z[ns] = self.rng.uniform(size=n_ns) < p

    def step(self):
        self._update_omega_b()
        self._update_tau()
        self._update_eps()
        self._update_eta()
        self._update_beta()
        self._update_omega_a()
        self._update_alpha()
        self._update_z()
