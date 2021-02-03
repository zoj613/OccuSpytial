import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri  # std norm cdf and its inverse

from ..distributions import precision_mvnorm

from .base import GibbsBase


def truncnorm_inf_ppf(a, p):
    """Return PPF from right hand tail of truncated normal distribution.

    Return values from the inteval (a, np.inf) for small |a|.
    """
    return -ndtri(ndtr(-a) * (1.0 - p))


def truncnorm_neginf_ppf(b, p):
    """Return PPF from left hand tail of truncated normal distribution.

    Return values from the inteval (-np.inf, b) for small |b|.
    """
    return ndtri(ndtr(b) * p)


class ProbitRSRGibbs(GibbsBase):
    r"""Sampler using probit link and RSR model for spatial random effects.

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
            - ``a_mu``: mean of the normal prior of detection covariates.
            - ``a_prec``: precision matrix of the normal prior of detection
              covariates.
            - ``b_mu``: mean of the normal prior of occupancy covariates.
            - ``b_prec``: precision matrix of the normal prior of occupancy
              covariates.
            - ``tau_rate``: rate parameter of the Gamma prior of the spatial
              parameter.
            - ``tau_shape``: shape parameter of the Gamma prior of the spatial
              parameter.
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
        w, v = np.linalg.eigh(omega)
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
        K = v[:, -self.fixed.q:]
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
        """Update the latent variable ``omega_a``.

        This variable is ssociated with the cofficients of the conditional
        detection covariates.
        """
        # get occopancy state of the sites where species was not observed.
        not_obs_occupancy = [i for i in self.fixed.not_obs if self.state.z[i]]
        self.state.exists = self.fixed.obs + not_obs_occupancy
        obs_mask = (self.y[self.state.exists] == 1)
        self.state.W = self.W[self.state.exists]
        loc = self.state.W @ self.state.alpha
        self.state.omega_a = np.zeros_like(loc)
        # sample from truncated normal distribution in the intervals (0, inf)
        # and (-inf, 0) using the inverse transform method
        # source: https://github.com/scipy/scipy/issues/12370
        a = loc[obs_mask]
        b = loc[~obs_mask]
        Ua = self.rng.random(size=a.shape[0])
        self.state.omega_a[obs_mask] = truncnorm_inf_ppf(-a, Ua) + a
        Ub = self.rng.random(size=b.shape[0])
        self.state.omega_a[~obs_mask] = truncnorm_neginf_ppf(-b, Ub) + b

    def _update_omega_b(self):
        """Update the latent variable ``omega_b``.

        This variable is associated with the cofficients of the occupancy
        covariates.
        """
        loc = self.X @ self.state.beta + self.state.spatial + self.state.eps
        exist_mask = (self.state.z == 1)
        a = loc[exist_mask]
        b = loc[~exist_mask]
        Ua = self.rng.random(size=a.shape[0])
        self.state.omega_b[exist_mask] = truncnorm_inf_ppf(-a, Ua) + a
        Ub = self.rng.random(size=b.shape[0])
        self.state.omega_b[~exist_mask] = truncnorm_neginf_ppf(-b, Ub) + b

    def _update_tau(self):
        eta = self.state.eta
        rate = 0.5 * eta @ self.fixed.Q @ eta + self.fixed.tau_rate
        self.state.tau = self.rng.gamma(self.fixed.tau_shape, 1 / rate)

    def _update_eps(self):
        mean = 0.5 * (
            self.state.omega_b - self.X @ self.state.beta - self.state.spatial
        )
        std = self.rng.standard_normal(mean.shape[0])
        self.state.eps = mean + self.fixed.eps_chol_factor * std

    def _update_eta(self):
        K = self.fixed.K
        eps = self.state.eps
        A = self.fixed.KTK + self.state.tau * self.fixed.Q
        b = K.T @ (self.state.omega_b - self.X @ self.state.beta - eps)
        self.state.eta = precision_mvnorm(b, A, self.rng)
        self.state.spatial = self.fixed.K @ self.state.eta

    def _update_alpha(self):
        WT = self.state.W.T
        A = WT @ WT.T + self.fixed.a_prec
        b = self.fixed.a_prec_by_mu + WT @ self.state.omega_a
        self.state.alpha = precision_mvnorm(b, A, self.rng)

    def _update_beta(self):
        b = self.fixed.b_prec_by_mu + self.X.T @ (
            self.state.omega_b - self.state.spatial - self.state.eps
        )
        self.state.beta = precision_mvnorm(
            b, self.fixed.XTX_plus_bprec, self.rng
        )

    def _update_z(self):
        no = self.fixed.not_obs
        ns = self.fixed.not_surveyed
        beta = self.state.beta
        K_eta = self.state.spatial

        num1 = ndtr(self.X[no] @ beta + K_eta[no] + self.state.eps[no])
        num2 = 1 - ndtr(self.fixed.W_not_obs @ self.state.alpha)
        stack_prod = np.multiply.reduceat(num2, self.fixed.stacked_w_indices)
        num = num1 * stack_prod
        p = num / ((1 - num1) + num)
        self.state.z[no] = self.rng.uniform(size=self.fixed.n_no) < p

        if ns:
            p = ndtr(self.X[ns] @ beta + K_eta[ns] + self.state.eps[ns])
            self.state.z[ns] = self.rng.uniform(size=self.fixed.n_ns) < p

    def step(self):
        self._update_omega_b()
        self._update_tau()
        self._update_eps()
        self._update_eta()
        self._update_beta()
        self._update_omega_a()
        self._update_alpha()
        self._update_z()
