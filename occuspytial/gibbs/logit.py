from math import sqrt

import numpy as np
from numpy.linalg import multi_dot
from polyagamma import random_polyagamma
from scipy.linalg import solve_triangular
from scipy.sparse import block_diag
from scipy.sparse.linalg import minres
from scipy.special import expit

from ..distributions import ensure_sums_to_zero, precision_mvnorm

from .base import GibbsBase


class _EtaICARPosterior:
    r"""Posterior distribution of the eta parameter of LogitICARGibbs.

    Parameters
    ----------
    Q : scipy sparse matrix
        ICAR precision matrix

    Methods
    -------
    rvs(b, omega, tau)

    Notes
    -----
    The distribution is a Multivariate Normal truncated on the hyperplane
    :math:`\mathbf{1}^T\mathbf{x} = \mathbf{0}`, and is of the form:

    .. math::

        \mathcal{N}(\mathbf{\Lambda}^{-1}\mathbf{b}, \mathbf{\Lambda}^{-1})


    where :math:`\mathbf{\Lambda} = (\tau \mathbf{Q} + \mathbf{S})`.

    This implementation allows one to sample from this normal distribution
    without explicitly factorizing :math:`\mathbf{\Lambda}` or computing its
    inverse. Since :math:`\mathbf{Q}` is known beforehand and can be factorized
    exactly once, and that :math:`\mathbf{S}` is diagonal, a random draw from
    this distribution can be computed efficiently as follows:

        1. Compute the square-root of :math:`\mathbf{S}` and multiply it by
           a standard normal draw.
        2. Muliply :math`\tau` by the pre-computed eigenfactor of
          :math:`\mathbf{Q}` and a standard normal draw of appropriate size.
        3. Sum the result of step 1) and 2) with :math:`\mathbf{b}`. The
           resulting array has the distribution

           .. math::

              \mathbf{y} = \mathcal{N}(\mathbf{b}, \mathbf{\Lambda})

        4. To get the draw from the desired distribution, we solve the linear
           system :math::`\mathbf{\Lambda}\mathbf{x} = \mathbf{y}` for
           :math:`\mathbf{x}`, and apply Algorithm 2 of [1]_ where the
           :math:`\mathbf{G}` in our case is :math:`\mathbf{1}^T` and
           :math:`\mathbf{r}` is :math:`\mathbf{0}`.
    """

    def __init__(self, Q):
        self._block_Q = block_diag((Q, Q), format='csc')
        s, u = np.linalg.eigh(Q.toarray())
        self._eigen = u[:, 1:] * np.sqrt(s[1:])
        self._n = Q.shape[0]
        self._rhs = np.ones(self._n * 2)
        self._n_plus_k = self._n + self._eigen.shape[1]
        self._guess = None

    def rvs(self, b, omega, tau, random_state):
        """Generate a random draw from this distribution."""
        eps = random_state.standard_normal(self._n_plus_k)
        rnorm1 = np.sqrt(omega) * eps[:self._n]
        rnorm2 = self._eigen @ (sqrt(tau) * eps[self._n:])
        out = b + rnorm1 + rnorm2

        block_prec = self._block_Q.copy()
        block_prec.data = tau * block_prec.data
        diag_vec = np.tile(omega, 2)
        block_prec.setdiag(block_prec.diagonal() + diag_vec)

        self._rhs[:self._n] = out
        # Iterative solvers are efficient at solving sparse large systems.
        xz, fail = minres(block_prec, self._rhs, x0=self._guess)
        # update starting guess for next call to the function
        self._guess = xz

        if fail:
            raise RuntimeError('MINRES solver did not converge!')

        x = xz[:self._n]
        z = xz[self._n:]

        ensure_sums_to_zero(x, z, out)

        return out


class LogitICARGibbs(GibbsBase):
    r"""Gibbs sample using logit link and ICAR model for spatial random effects.

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
    pertub : float, optional
        The value by which to pertube the diagonal of the spatial precision
        matrix in order to stabilize the cholesky factorization of the
        posterior precision matrix of the :math:`\\eta` parameter. This is
        sometimes referred to as modified cholesky decompostion [1]_ . It helps
        get numerically stable samples from the conditional distribution of
        this parameter.

    Methods
    -------
    sample(size, burnin=0, start=None, chains=1, progressbar=True)

    See Also
    --------
    occuspytial.gibbs.probit.ProbitRSRGibbs :
        A gibbs sampler using a probit link function

    Notes
    -----
    The algorithm developed here is the same as the one presented in [2]_
    except the model used to account for spatial correlation is the
    Intrinsic Conditional Autoregressive (ICAR) model. A polya-gamma random
    variable distribution is used in a data augmentation strategy in order to
    obtain known conditional distributions to sample from, thus arising the
    Gibbs sampler impelemented here. Details of the sampler are explained in
    [2]_ .

    References
    ----------
    .. [1] McSweeney, T. Modified Cholesky Decomposition and
       Applications. 2017; Masters thesis, University of Manchester.
    .. [2]  Clark, AE, Altwegg, R. Efficient Bayesian analysis of occupancy
       models with logit link functions. Ecol Evol. 2019; 9: 756– 768.
       https://doi.org/10.1002/ece3.4850.
    """

    def __init__(self, Q, W, X, y, hparams=None, random_state=None):
        super().__init__(Q, W, X, y, hparams, random_state)
        self._configure(Q, hparams)

    def _configure(self, Q, hparams):
        super()._configure(Q, hparams)
        self.dists.eta_posterior = _EtaICARPosterior(self.fixed.Q)

    def _update_omega_a(self):
        """Update the latent variable ``omega_a``.

        This variable is ssociated with the cofficients of the conditional
        detection covariates.
        """
        # get occupancy state of the sites where species was not observed.
        not_obs_occupancy = [i for i in self.fixed.not_obs if self.state.z[i]]
        self.state.exists = self.fixed.obs + not_obs_occupancy
        self.state.W = self.W[self.state.exists]
        b = self.state.W @ self.state.alpha
        self.state.omega_a = random_polyagamma(
            1, b, disable_checks=True, random_state=self.rng
        )

    def _update_omega_b(self):
        """Update the latent variable ``omega_b``.

        This variable is associated with the cofficients of the occupancy
        covariates.
        """
        b = self.X @ self.state.beta + self.state.spatial
        self.state.omega_b = random_polyagamma(
            1, b, disable_checks=True, random_state=self.rng
        )

    def _update_tau(self):
        eta = self.state.eta
        rate = 0.5 * (eta @ self.fixed.Q @ eta) + self.fixed.tau_rate
        self.state.tau = self.rng.gamma(self.fixed.tau_shape, 1 / rate)

    def _update_eta(self):
        omega = self.state.omega_b
        b = self.state.k - (omega * (self.X @ self.state.beta))
        self.state.eta = self.dists.eta_posterior.rvs(
            b, omega, self.state.tau, self.rng
        )
        self.state.spatial = self.state.eta

    def _update_alpha(self):
        WT = self.state.W.T
        y = self.y[self.state.exists] - 0.5
        A = (WT * self.state.omega_a) @ WT.T + self.fixed.a_prec
        b = WT @ y + self.fixed.a_prec_by_mu
        self.state.alpha = precision_mvnorm(b, A, random_state=self.rng)

    def _update_beta(self):
        spat = self.state.spatial
        omega = self.state.omega_b
        XT = self.X.T
        A = (XT * omega) @ self.X + self.fixed.b_prec
        b = XT @ (self.state.k - (omega * spat)) + self.fixed.b_prec_by_mu
        self.state.beta = precision_mvnorm(b, A, random_state=self.rng)

    def _update_z(self):
        """Update the occupancy state of each site."""
        no = self.fixed.not_obs
        ns = self.fixed.not_surveyed
        beta = self.state.beta
        spat = self.state.spatial

        num1 = expit(self.X[no] @ beta + spat[no])
        num2 = expit(self.fixed.W_not_obs @ -self.state.alpha)
        stack_prod = np.multiply.reduceat(num2, self.fixed.stacked_w_indices)
        num = num1 * stack_prod
        p = num / ((1 - num1) + num)
        self.state.z[no] = self.rng.uniform(size=self.fixed.n_no) < p

        if ns:
            p = expit(self.X[ns] @ beta + spat[ns])
            self.state.z[ns] = self.rng.uniform(size=self.fixed.n_ns) < p

        self.state.k = self.state.z - 0.5

    def step(self):
        """Complete one gibbs sampler update.

        The method should not be called directly. It is called internally
        by the ``sample`` method.
        """
        self._update_omega_b()
        self._update_tau()
        self._update_eta()
        self._update_beta()
        self._update_omega_a()
        self._update_alpha()
        self._update_z()


class _EtaRSRPosterior:
    r"""Posterior distribution of the eta parameter of LogitRSRGibbs.

    Parameters
    ----------
    Q : np.ndarray
        RSR precision matrix.

    Methods
    -------
    rvs(b, omega, tau)

    Notes
    -----
    The distribution is of the form:

    .. math::

        \mathcal{N}(\mathbf{\Lambda}^{-1}\mathbf{b}, \mathbf{\Lambda}^{-1})

    where
    .. math::

      \mathbf{\Lambda} = (\tau \mathbf{Q}+\mathbf{K}^{-1}\mathbf{S}\mathbf{K})

    This implementation allows one to sample from this normal distribution
    without explicitly factorizing :math:`\mathbf{\Lambda}` or computing its
    inverse. Since :math:`\mathbf{Q}` and :math:`\mathbf{K}` are known
    beforehand and can be factorized exactly once, and that :math:`\mathbf{S}`
    is diagonal, a random draw from this distribution can be computed
    efficiently as follows:

        1. Multiply :math:`\mathbf{K}^{-1}` with the square-root of
           :math:`\mathbf{S}` and a standard normal draw.
        2. Muliply :math`\tau` by the pre-computed eigenfactor of
          :math:`\mathbf{Q}` and a standard normal draw of appropriate size.
        3. Sum the result of step 1) and 2) with :math:`\mathbf{b}`. The
           resulting array has the distribution

           .. math::

              \mathbf{y} = \mathcal{N}(\mathbf{b}, \mathbf{\Lambda})

        4. To get the draw from the desired distribution, we solve the linear
           system :math::`\mathbf{\Lambda}\mathbf{x} = \mathbf{y}` for
           :math:`\mathbf{x}`
    """

    def __init__(self, Q, K):
        s, u = np.linalg.eigh(Q)
        self._Q = Q
        self._eigen = u * np.sqrt(s)
        self._KT = K.T
        self._n = Q.shape[0]
        self._n_plus_k = self._n + K.shape[0]

    def rvs(self, b, omega, tau, random_state):
        """Generate a random draw from this distribution."""
        factor1 = self._KT * np.sqrt(omega)
        factor2 = sqrt(tau) * self._eigen

        eps = random_state.standard_normal(self._n_plus_k)
        rnorm1 = factor1 @ eps[self._n:]
        rnorm2 = factor2 @ eps[:self._n]
        out = b + rnorm1 + rnorm2

        prec = factor1 @ factor1.T + tau * self._Q
        out = np.linalg.solve(prec, out)
        return out


class LogitRSRGibbs(LogitICARGibbs):
    """Gibbs sampler using logit link and RSR model for spatial random effects.

    This algorithm is an implementation of the gibbs sampler in [1]_ where a
    Reduced Spatial Regression (RSR) model is used to account for spatial
    autocorrelation in a single-season site occupancy model.

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
    occuspytial.gibbs.logiit.LogitICARGibbs :
        The same sampler but using an ICAR model.
    occuspytial.gibbs.probit.ProbitRSRGibbs :
        A gibbs sampler using a probit link function.

    References
    ----------
    .. [1]  Clark, AE, Altwegg, R. Efficient Bayesian analysis of occupancy
       models with logit link functions. Ecol Evol. 2019; 9: 756– 768.
       https://doi.org/10.1002/ece3.4850.
    """

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
        # replace Q with Minv
        Q_copy = self.fixed.Q
        del self.fixed.Q
        self.fixed.Q = K.T @ Q_copy @ K
        self.fixed.K = K

        if not hparams:
            # `_set_default_hyperparams` has been called so modify tau_shape
            del self.fixed.tau_shape
            self.fixed.tau_shape = 0.5 + 0.5 * self.fixed.q

        del self.dists.eta_posterior
        self.dists.eta_posterior = _EtaRSRPosterior(self.fixed.Q, K)

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
        b = K.T @ (self.state.k - omega * (self.X @ self.state.beta))
        self.state.eta = self.dists.eta_posterior.rvs(
            b, omega, self.state.tau, self.rng
        )
        self.state.spatial = K @ self.state.eta
