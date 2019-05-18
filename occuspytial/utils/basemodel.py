from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from .utils import CustomDict


class MCMCModelBase(ABC):

    def __init__(
            self,
            X: np.ndarray,
            W: Dict[int, np.ndarray],
            y: Dict[int, np.ndarray],
            init: Dict[str, Tuple[np.ndarray, float]],
            hypers: Dict[str, Tuple[np.ndarray, float]]
    ) -> None:

        self.W = W
        self.X = X
        self.y = y
        self._n = X.shape[0]  # number of sites in total
        self._s = len(self.y)  # number of surveyed sites
        self._us = self._n - self._s  # number of unsurveyed sites
        self.Xs = X[:self._s]  # X sub-matrix for surveyed sites.
        self._V = np.array(
            [self.W[i].shape[0] for i in range(self._s)],
            dtype=np.int64
        )
        self.init = init
        self.hypers = hypers
        self._not_obs: List[int] = []  # surveyed sites where not observed
        # initialize z, the site occupancy state
        self._z = np.ones(self._n, dtype=np.int64)
        for i in range(self._s):
            if not any(self.y[i]):
                self._not_obs.append(i)
                self._z[i] = 0.0
        self.not_obs: np.ndarray = np.array(self._not_obs, dtype=np.int64)
        # array to store prob updates for sites where species is not obversed
        self._probs = np.zeros(self.not_obs.size, dtype=np.float64)
        # array to store occupancy prob for sites where species is unsurveyed
        self._s_probs = np.zeros(self._s, dtype=np.float64)
        self._us_probs = np.zeros(self._us, dtype=np.float64)
        # stacked W matrix for all sites where species is not observed
        self._W_ = CustomDict(self.W).slice(self.not_obs)

        self._names: List[str] = []
        for i in range(W[0].shape[1]):
            self._names.append(r"$\alpha_{0}$".format(i))
        for i in range(X.shape[1]):
            self._names.append(r"$\beta_{0}$".format(i))
        # specify the names of the posterior parameters
        self._names.append("PAO")
        self._names.append(r"$\tau$")
        self._alpha = init["alpha"]
        self._beta = init["beta"]
        self._tau = init["tau"]
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
