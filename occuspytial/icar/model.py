import logging
from typing import Callable, Dict, Optional

import numpy as np  # type: ignore
from numpy.linalg import multi_dot
from scipy.linalg import eigh, inv, solve_triangular as tri_solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import expit  # inverse logit
from pypolyagamma import PyPolyaGamma

from occuspytial.icar.helpers.ctypesfunc import num_prod
from occuspytial.utils.misc import CustomDict
from occuspytial.utils.basemodel import MCMCModelBase, ParamType
from occuspytial.utils.stats import affine_sample
from occuspytial.utils.visualization import ProgressBar

logger = logging.getLogger(__name__)
pg = PyPolyaGamma()


class RSR:

    def __init__(
        self,
        X: np.ndarray,
        Q: csc_matrix,
        init: ParamType,
        hypers: ParamType,
        threshold: float,
        num_of_surveyed_sites: int
    ) -> None:

        n = X.shape[0]
        XTX_i = inv(X.T @ X, overwrite_a=True, check_finite=False)
        P = -multi_dot([X, XTX_i, X.T])
        P[np.diag_indices(n)] += 1
        # above 2 lines equivalent to: I - X @ XTX @ XT
        A = Q.copy()
        A.data = -A.data
        A.setdiag(np.zeros(n))
        omega = n * (P @ A @ P) / A.sum()
        # return eigen vectors of omega and keep first q columns
        # corresponding to the q largest eigenvalues of omega greater
        # than the specified threshold
        w, v = eigh(omega, overwrite_a=True, check_finite=False)
        w, v = w[::-1], np.fliplr(v)  # order eigens in descending order
        _msg = "threshold value needs to be between 0 and 1 (inclusive)"
        if not 0 <= threshold <= 1:
            logger.error(_msg)
            raise Exception(_msg)
        q = len(w[w >= threshold])  # number of eigenvalues > threshold
        _msg = "Threshold is set too high. Please lower it."
        if not q > 0:
            logger.error(_msg)
            raise Exception(_msg)
        logger.info("dimension reduced: {0}->{1}".format(n, q))
        self._K = v[:, :q]  # keep first q eigenvectors of ordered eigens
        self.Minv = self._K.T @ Q @ self._K
        self._Ks = self._K[:num_of_surveyed_sites]
        self.Q = self.Minv  # replace Q with Minv in the underlying ICAR
        self._shape = hypers["shape"] + 0.5 * q
        # theta initial values
        self._theta = init.get("theta", np.random.normal(0, 1, q))


class ICAR(MCMCModelBase):

    def __init__(
            self,
            X: np.ndarray,
            W: Dict[int, np.ndarray],
            y: Dict[int, np.ndarray],
            Q: np.ndarray,
            init: ParamType,
            hypers: ParamType,
            use_rsr: bool = False,
            threshold: float = 0.5
    ) -> None:

        super().__init__(X, W, y, init, hypers)
        rank = np.linalg.matrix_rank(Q)
        self.Q = csc_matrix(Q)
        if rank >= self._n:
            _msg = "Q must be a singular matrix. Cannot continue."
            logger.error(_msg)
            raise Exception(_msg)
        self._k = self._z - 0.5  # initial value of k = z - 0.5*1
        self._Wc = None
        self._kc = None
        # constant shape parameter for _tau_update method
        self._shape = self.hypers["shape"] + 0.5 * rank
        # attributes specific to omega_b update method
        self._omega_a = None
        self._omega_b = np.empty(self._n)
        self._ones = np.ones_like(self._omega_b)
        self._use_rsr = use_rsr
        if use_rsr:
            # RSR model specific attributes
            rsr = RSR(X, self.Q, self.init, self.hypers, threshold, self._s)
            self.__dict__.update(rsr.__dict__)
        else:
            # ICAR model specific attributes
            self._eta = self.init["eta"]

    def _omega_a_update(self) -> None:

        Wk_alpha = self._Wc.T @ self._alpha
        self._omega_a = np.empty_like(Wk_alpha)
        pg.pgdrawv(np.ones_like(self._omega_a), Wk_alpha, self._omega_a)

    def _omega_b_update(
            self,
            vec: np.ndarray,
            nonspat: bool = False
    ) -> None:

        if nonspat:
            pg.pgdrawv(self._ones, self.X @ self._beta, self._omega_b)
        else:
            pg.pgdrawv(self._ones, self.X @ self._beta + vec, self._omega_b)

    def _tau_update(self, vec: np.ndarray) -> None:

        rate = 0.5 * vec @ self.Q @ vec + self.hypers["rate"]
        self._tau = np.random.gamma(shape=self._shape, scale=1 / rate)

    def _eta_update(self) -> None:
        # Cong et al method
        prec = self.Q.copy()
        prec.data = prec.data * self._tau
        prec.setdiag(prec.diagonal() + self._omega_b)
        b = self._k - self._omega_b * (self.X @ self._beta)
        s, sp_chol_factor = affine_sample(b, prec, return_factor=True)

        try:
            rhs = np.column_stack((s, self._ones))
            xz = sp_chol_factor.solve_A(rhs)
            logger.debug("updated eta using scikit-sparse routines.")
        except AttributeError:

            xz = splu(
                prec,
                permc_spec='MMD_AT_PLUS_A',
                options=dict(SymmetricMode=True)
            ).solve(rhs)
            logger.debug("updated eta using scipy routines.")
        finally:
            a = xz.sum(axis=0)
            a = -a[0] / a[1]
            self._eta = xz[:, 0] + a * xz[:, 1]

    def _theta_update(self) -> None:

        k = self._k[:self._s]
        omega = self._omega_b[:self._s]
        prec = self._tau * self.Minv + (self._Ks.T * omega) @ self._Ks
        b = self._Ks.T @ (k - omega * (self.Xs @ self._beta))
        prec_theta, upper_tri = affine_sample(b, prec, return_factor=True)
        x = tri_solve(upper_tri, prec_theta, trans=1)
        self._theta = tri_solve(upper_tri, x)

    def _wk_update(self) -> None:

        z_ind = np.where(self._z[:self._s] == 1)[0]
        self._Wc = CustomDict(self.W).slice(z_ind).T
        self._kc = CustomDict(self.y).slice(z_ind) - 0.5

    def _alpha_update(self) -> None:

        a_mu, a_prec = self.hypers["a_mu"], self.hypers["a_prec"]
        prec = (self._Wc * self._omega_a) @ self._Wc.T + a_prec
        b = a_prec @ a_mu + self._Wc @ self._kc
        prec_alpha, upper_tri = affine_sample(b, prec, return_factor=True)
        x = tri_solve(upper_tri, prec_alpha, trans=1)
        self._alpha = tri_solve(upper_tri, x)

    def _beta_update(self, vec: np.ndarray, nonspat: bool = False) -> None:

        k = self._k[:self._s]
        omega = self._omega_b[:self._s]
        b_mu, b_prec = self.hypers["b_mu"], self.hypers["b_prec"]
        prec = (self.Xs.T * omega) @ self.Xs + b_prec
        if nonspat:
            b = self.Xs.T @ k + b_prec @ b_mu
        else:
            vec = vec[:self._s]
            b = self.Xs.T @ (k - omega * vec) + b_prec @ b_mu
        prec_beta, upper_tri = affine_sample(b, prec, return_factor=True)
        x = tri_solve(upper_tri, prec_beta, trans=1)
        self._beta = tri_solve(upper_tri, x)

    def _z_update(self, vec: np.ndarray, nonspat: bool = False) -> None:

        if nonspat:
            occ = expit(self.X[self.not_obs] @ self._beta)
        else:
            occ = expit(self.X[self.not_obs] @ self._beta + vec[self.not_obs])

        omd = expit(self._W_ @ -self._alpha)
        # calculate the numerator product for each site i
        # in the expression for the probability of z=1 | y=0
        # num_prod stores the product on the object self._probs.
        self._probs = occ.copy()
        num_prod(
            omd, self.not_obs, self.not_obs.shape[0], self._V, self._probs
        )
        # use the stored values to calculate the probability
        self._probs /= (1 - occ + self._probs)
        # update the occupancy state array by sampling from a Bernoulli with \
        # self._prob as its parameter value.
        self._z[self.not_obs] = np.random.binomial(n=1, p=self._probs)

        # sampling for sites not surveyed
        if self._us > 0:
            # calc the probability of z = 1 | site i not surveyed
            if nonspat:
                self._us_probs = expit(self.X[-self._us:] @ self._beta)
            else:
                self._us_probs = expit(
                    self.X[-self._us:] @ self._beta + vec[-self._us:]
                )
            # update occupancy state by sampling from a Bernoulli(us_prob)
            self._z[-self._us:] = np.random.binomial(n=1, p=self._us_probs)
        self._k = self._z - 0.5  # update k = _z - 0.5 for next iteration

    def _update_func(
            self,
            func: Callable[[], None],
            x: np.ndarray,
            y: Optional[np.ndarray] = None
    ) -> None:

        if y is not None:
            args = [x @ y, y]
            self._theta_update()
        else:
            args = [x]
        self._omega_b_update(args[0])
        func()  # either self._eta or self._theta
        self._tau_update(args[-1])
        self._z_update(args[0])
        self._beta_update(args[0])
        self._wk_update()
        self._omega_a_update()
        self._alpha_update()

    def _update_func_nonspat(self) -> None:

        self._omega_b_update(None, nonspat=True)
        self._z_update(None, nonspat=True)
        self._beta_update(None, nonspat=True)
        self._wk_update()
        self._omega_a_update()
        self._alpha_update()

    def _params_update(self, nonspat: bool = False) -> None:
        """ Function doc """
        if nonspat:
            self._update_func_nonspat()
        elif self._use_rsr:
            self._update_func(self._theta_update, self._K, self._theta)
        else:
            self._update_func(self._eta_update, self._eta)

    def _new_init(self, init: ParamType) -> None:

        self._alpha = init["alpha"]
        self._beta = init["beta"]
        self._tau = init["tau"]
        if self._use_rsr:
            default_value = np.random.normal(0, 1, self.Minv.shape[0])
            self._theta = init.get("theta", default_value)
        else:
            self._eta = init["eta"]

    def run_sampler(
            self,
            iters: int = 2000,
            burnin: Optional[int] = None,
            init: Optional[ParamType] = {},
            progressbar: bool = True,
            nonspatial: bool = False
    ) -> None:

        if init != {}:
            logger.info("supplying model with parameter initial values.")
            self._new_init(init)

        if burnin is None:
            logger.debug("burnin value not set, using the default value.")
            burnin = int(0.5 * iters)  # default burnin value

        post_burnin: int = iters - burnin
        if post_burnin <= 0:
            _msg = "iters needs to be larger than burnin. Sampling failed."
            logger.error(_msg)
            raise ValueError(_msg)

        if progressbar:
            logger.debug("turning on progress bar...")
            bar = ProgressBar(iters)

        setattr(self, '_traces', np.empty((post_burnin, len(self._names))))
        setattr(self, 'z_mat', np.empty((post_burnin, self._n)))

        if self._use_rsr:
            setattr(
                self, 'theta_mat', np.empty((post_burnin, self._K.shape[1]))
            )

        j = 0
        logger.debug("beginning the first iteration...")
        for i in range(iters):
            self._params_update(nonspatial)
            if progressbar:
                bar.update()
            if i >= burnin:
                self._traces[j] = np.concatenate((
                    self._alpha,
                    self._beta,
                    np.array([self._z.mean(), self._tau])
                ))
                self.z_mat[j] = self._z
                if self._use_rsr and not nonspatial:
                    self.theta_mat[j] = self._theta

                j += 1
        if progressbar:
            bar._FILE.close()
