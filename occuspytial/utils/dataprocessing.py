from operator import itemgetter
from typing import Dict, Iterable

import numpy as np


class DetectionDataObject:
    """A class serving as a container for Detection data W and y.

    W is a dictionary with keys representing site numbers and values
    representing the corresponding design matrix for that site. The `y`
    dictionary has the same keys as W but its values are 1-D arrays
    representing the detection/non-detection data of each site during
    each visit. Only one of W and y can be used to instantiate this class

    Args:
        data (Dict[int, np.ndarray]): A dictionary object containing
            detection data of each surveyed site.

    Attribute:
        visits_per_sites (np.ndarray): An array containing the number of
            visits per site.
    """
    def __init__(self, data: Dict[int, np.ndarray]) -> None:
        self.data = data

    def __getitem__(self, sites) -> np.ndarray:
        """Get concatenated value(s) of ``self.data`` corresponding to
        the key(s) provided in ``sites``.
        """
        if isinstance(sites, int):
            return self.data[sites]
        if isinstance(sites, Iterable):
            out = itemgetter(*sites)(self.data)
            return np.concatenate(out)

    def __len__(self):
        return len(self.data)

    @property
    def visits_per_site(self) -> np.ndarray:
        """Get the number of visits per surveyed site."""
        surveyed_sites = range(len(self))
        visits = tuple(self.data[i].shape[0] for i in surveyed_sites)
        return np.array(visits, dtype=np.int64)


class HyperParams:
    """Container class for hyper-parameter values.

    Args:
        hypers (ParamType): The hyperparameters of the model.
        X (np.ndarray): Design matrix of the occupancy process.
        W (Dict[int, np.ndarray]): Design matrices of the detection
            detection process.
    """
    def __init__(self, hyper, X: np.ndarray, W: Dict[int, np.ndarray]) -> None:
        if hyper is None:
            num_x_cols = X.shape[1]
            num_w_cols = W[0].shape[1]
            defaults = dict(
                alpha_mu=np.zeros(num_w_cols),
                alpha_prec=np.diag([1. / 1000] * num_w_cols),
                beta_mu=np.zeros(num_x_cols),
                beta_prec=np.diag([1. / 1000] * num_x_cols),
                shape=0.5,
                rate=0.0005
            )
            self.__dict__.update(defaults)
        else:
            self.__dict__.update(hyper)


class InitValues:
    """Container class for parameter initial values.

    Args:
        hypers (ParamType): The hyperparameters of the model.
        X (np.ndarray): Design matrix of the occupancy process.
        W (Dict[int, np.ndarray]): Design matrices of the detection
            detection process.
    """
    def __init__(self, inits, X: np.ndarray, W: Dict[int, np.ndarray]) -> None:
        if inits is None:
            num_x_cols = X.shape[1]
            num_w_cols = W[0].shape[1]
            total_sites = X.shape[0]
            defaults = dict(
                alpha=np.zeros(num_w_cols),
                beta=np.zeros(num_x_cols),
                tau=10.,
                eta=np.random.uniform(-10, 10, size=total_sites)
            )
            self.__dict__.update(defaults)
        else:
            self.__dict__.update(inits)
