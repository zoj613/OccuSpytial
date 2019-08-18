from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from occuspytial.utils.dataprocessing import DetectionDataObject
ParamType = Dict[str, Union[np.ndarray, float]]


def _set_param_values(X, W, hypers, inits) -> Tuple[ParamType, ParamType]:
    """Set default parameter values if none are provided by the user"""
    hyper_out = {}
    init_out = {}
    p = X.shape[1]
    p_v = W[0].shape[1]
    n = X.shape[0]
    hyper_out["a_mu"] = hypers.get("a_mu", np.zeros(p_v))
    hyper_out["a_prec"] = hypers.get("a_prec", np.diag([1. / 1000] * p_v))
    hyper_out["b_mu"] = hypers.get("b_mu", np.zeros(p))
    hyper_out["b_prec"] = hypers.get("b_prec", np.diag([1. / 1000] * p))
    hyper_out["shape"] = hypers.get("shape", 0.5)
    hyper_out["rate"] = hypers.get("rate", 0.0005)

    init_out["alpha"] = inits.get("alpha", np.zeros_like(hyper_out["a_mu"]))
    init_out["beta"] = inits.get("beta", np.zeros_like(hyper_out["b_mu"]))
    init_out["tau"] = inits.get("tau", 10)
    init_out["eta"] = inits.get("eta", np.random.uniform(-10, 10, size=n))

    return hyper_out, init_out


class MCMCModelBase(ABC):
    """A base class for instantiating a site occupancy model.

    Attributes:
        W (Dict[int, np.ndarray]): The detection process design matrices
            for each site stored as a dictionary. The key is the site
            number while the value is the design matrix of that site
            stored as a 2D numpy array.
        X (np.ndarray): The occupancy process design matrix.
        y (np.ndarray): The detection/non-detection data for each site
            stored as a dictionary. The key is the site number and the
            value of each key is a 1D numpy array whose length is the
            number of time the site was visited. The value are 1's (if
            a species was observed on that particular visit) and 0's (
            if a species was not observed).
        Xs (np.ndarray): A subset of the occupancy process design matrix
            for sites that were surveyed out of the total.
        init (ParamType): The initial values of model parameters.
        hypers (ParamType): The hyperparameters of the model.
        not_obs (np.ndarray): A 1D array whose entries are the site
            numbers where the species was not observed at all on any
            visit.
        avg_occ_probs (np.ndarray): The average occupancy probability of
            each site.

    Methods:
        run_sampler: An abstract method to define the posterior sampling
        process of a Bayesian model.
    """
    def __init__(
            self,
            X: np.ndarray,
            W: Dict[int, np.ndarray],
            y: Dict[int, np.ndarray],
            init: Optional[ParamType] = {},
            hypers: Optional[ParamType] = {}
    ) -> None:

        self.W = DetectionDataObject(W)
        self.X = X
        self.y = DetectionDataObject(y)
        self._n = X.shape[0]  # number of sites in total
        self._s = len(self.y)  # number of surveyed sites
        self._us = self._n - self._s  # number of unsurveyed sites
        self.Xs = X[:self._s]  # X sub-matrix for surveyed sites.
        self._V = self.W.visits_per_site
        self.hypers, self.init = _set_param_values(X, W, hypers, init)

        self._not_obs: List[int] = []  # surveyed sites where not observed
        # initialize z, the site occupancy state
        self._z = np.ones(self._n, dtype=int)
        for i in range(self._s):
            if not any(self.y[i]):
                self._not_obs.append(i)
                self._z[i] = 0
        self.not_obs: np.ndarray = np.array(self._not_obs, dtype=int)
        # array to store prob updates for sites where species is not obversed
        self._probs = np.zeros(self.not_obs.size, dtype=float)
        # array to store occupancy prob for sites where species is unsurveyed
        self._us_probs = np.zeros(self._us, dtype=float)
        # stacked W matrix for all sites where species is not observed
        self._W_ = self.W[tuple(self.not_obs)]
        # store parameter names to be used in plot labels.
        self._names: List[str] = []
        for i in range(W[0].shape[1]):
            self._names.append(r"$\alpha_{0}$".format(i))
        for i in range(X.shape[1]):
            self._names.append(r"$\beta_{0}$".format(i))
        # specify the names of the posterior parameters
        self._names.append("PAO")
        self._names.append(r"$\tau$")
        self._alpha = self.init["alpha"]  # inital values for alpha
        self._beta = self.init["beta"]  # initial values for beta
        self._tau = self.init["tau"]  # initial values for tau
        self.avg_occ_probs = np.ones(self._n)

    @abstractmethod
    def _alpha_update(self) -> None:
        pass

    @abstractmethod
    def _beta_update(self) -> None:
        pass

    @abstractmethod
    def _z_update(self) -> None:
        pass

    @abstractmethod
    def run_sampler(self, iters: int = 2000) -> None:
        pass
