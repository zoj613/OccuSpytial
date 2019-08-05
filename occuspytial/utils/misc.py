import logging
import logging.config
import os
from pathlib import Path

from typing import Any, List, Optional, Sequence, Tuple
import warnings

import numpy as np  # type: ignore

try:
    import toml
except ImportError:
    warnings.showwarning(
        "The toml package is not installed. Logging configuration through "
        "the toml config file can not be used until it is installed.",
        category=ImportWarning,
        filename=__name__,
        lineno=27
    )

logger = logging.getLogger(__name__)


class CustomDict(dict):
    """
    A custom dictionary that supports indexing via its keys. The index
    can be *args or any iterable e.g., numpy array, list, tuple.
    """
    # a mix-in class for group-indexing W and y dictionaries
    def slice(self, *keys: Sequence[Any]) -> np.ndarray:
        """Take in a sequence of keys as input and return the corres-
        ponding values as one concatenated numpy array over axis 0.

        Args:
            *keys (Sequence[Any]): variable length sequence of keys.

        Returns:
            np.ndarray: a stacked numpy array of values that correspond
            to the key arguments. If the values of numpy arrays then
            they are stacked row-wise.
        """
        try:
            out = [self[k] for k in keys]
        except TypeError:  # if input is not hashable
            out = [self[k] for k in keys[0]]
        return np.concatenate(tuple(out))


class SpatialStructure:
    """ A class intended for generating spatial precision matrix used in
    models like CAR, ICAR and RSR.

    Args:
        n (int): The number of sites.

    Attributes:
        n (int): The number of sites.

    Methods:
        spatial_precision: Returns a randomly generated spatial presicion
            matrix of size n by n.
    """

    def __init__(self, n: int) -> None:

        self.n = n

    def _generate_random_lattice(
            self,
            n: Optional[int] = None,
            fix_square: bool = False
    ) -> None:
        """Generates a random lattice using n number of sites. The lattice
        can either be square or rectangular.

        Args:
            n (Optional[int]): Number of sites. Defaults to None.
            fix_square (bool, optional): Whether or not the lattice
                should be square. A false value will return a rectangular
                lattice. Defaults to False.
        """
        if n is not None:
            a = n
        else:
            a = self.n
        if fix_square:
            _sqrt = int(np.sqrt(a))
            factors = (_sqrt, _sqrt)
        else:
            b = np.arange(1, a + 1)
            c = b[a % b == 0][1:-1]  # multiplicative factors of n except for 1
            d = []
            for i in c:
                out = c[i * c == a][0]  # factor whose product with i equals a
                d.append((i, out))

            d = d[-(a // 2):]  # remove duplicate pairs
            # randomly select one element
            factors = d[np.random.randint(0, len(d))]
        # create a lattice rectangular grid of dims factors[0] x factors[1]
        self.lattice = np.arange(1, a + 1).reshape(factors)

    def _neighbor_indx(
            self,
            indx: Tuple[int],
            n_type: int = 4
    ) -> List[Tuple[int]]:
        """Given an integer index of a site on a lattice grid, return
        site's neighbors' indices. The neighborhood structure can either
        be 4 neighbors or 8 neighbors.

        Args:
            indx (Tuple[int]): The current sites's index as a tuple (x, y)
            n_type (int, optional): Number of neighbors. Defaults to 4.

        Raises:
            Exception: if the number of neighbors supplied as input is
                not one of 4 and 8.

        Returns:
            List[Tuple[int]]: A list of tuple indices for each of the
            n_type neighbors of the current site.
        """
        if n_type not in [4, 8]:
            msg = "The number of neighbors specified must be one of [4, 8]"
            logger.error(msg)
            raise Exception(msg)

        out = []
        if n_type == 8:
            out.append((indx[0] - 1, indx[1] - 1))  # north west
            out.append((indx[0] + 1, indx[1] - 1))  # south west
            out.append((indx[0] - 1, indx[1] + 1))  # north east
            out.append((indx[0] + 1, indx[1] + 1))  # south east

        out.append((indx[0], indx[1] - 1))
        out.append((indx[0], indx[1] + 1))
        out.append((indx[0] - 1, indx[1]))
        out.append((indx[0] + 1, indx[1]))

        # taking care of edge cases and making sure the neighbours are
        # all valid indices that fall within the generated lattice grid.
        edge_case = (
            indx[0] == 0  # current site is in first row of lattice
            or indx[1] == 0  # is in first column of lattice
            or indx[0] == self.lattice.shape[0] - 1  # is in last row
            or indx[1] == self.lattice.shape[1] - 1  # is in last column
        )
        if edge_case:
            logger.debug(f"site index {indx} is an edge case")
            for item in np.array(out):
                if (
                    np.any(item < 0)
                    or item[0] >= self.lattice.shape[0]
                    or item[1] >= self.lattice.shape[1]
                ):
                    out.remove(tuple(item))
                    logger.debug(f"removed {item} as a neighbor of {indx}")
            logger.debug(f"index neighbor list is: {out}")
        return out

    def _adjacency_matrix(self, n_type: int = 48) -> None:
        """ use the generated lattice to create an adjacency matrix A, where
        an element A[i, j] = 1 if i and j are neighbors and A[i, j] = 0
        otherwise for i =/= j. A[i, i] = 0 for all i.

        Args:
            n_type (int): The number of neighbors. Input should be one
                of [4, 8, 48]. 4 = Four neighbours, 8 = Eight neighbors,
                and 48 = either of the two (selected randomly).

        Raises:
            IndexError: if another element is not one of the current
                element's neighbors.
        """
        a = np.zeros((self.n, self.n))
        for indx, site in np.ndenumerate(self.lattice):
            a[site - 1, site - 1] = 0
            # randomly decide the maximum number of neighbors for site.
            if n_type == 48:
                type_of_neighbor = np.random.choice([4, 8], p=[0.5, 0.5])
                neighbor_indx = self._neighbor_indx(indx, type_of_neighbor)
            else:
                neighbor_indx = self._neighbor_indx(indx, n_type)

            for row, col in neighbor_indx:
                neighbor_site = self.lattice[row, col]
                a[site - 1, neighbor_site - 1] = 1
                a[neighbor_site - 1, site - 1] = 1

        self.A = a

    def spatial_precision(
            self,
            n_type: int = 48,
            rho: float = 1.,
            square_lattice: bool = False
    ) -> np.ndarray:
        """Randomly enerates a spatial precision matrix of size n by n,
        as described in the CAR and ICAR models.

        Args:
            n_type (int, optional): Neighborhood structure. Input should
                be one of [4, 8, 48]. 4 = Four neighbours, 8 = Eight
                neighbors, and 48 = either of the two (selected randomly).
                Defaults to 48.
            rho (int, optional): Set to 1 if creating a precision matrix
                for the ICAR model, else set it to a float between 0 and
                1. Defaults to 1.
            square_lattice (bool, optional): Whether or not the lattice
                should be square. A false value will return a rectangular
                lattice. Defaults to False.

        Returns:
            np.ndarray: A spatial precision matrix of size n by n.
        """
        self._generate_random_lattice(fix_square=square_lattice)
        self._adjacency_matrix(n_type)
        d_mat = np.diag(self.A.sum(axis=0))
        return d_mat - rho * self.A


# adapated from:
# https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
def log_config(
    default_path: str = 'log_config.toml',
    default_level: int = logging.DEBUG,
    env_key: str = 'LOG_CFG'
):
    """Setup logging configuration"""
    path = Path(default_path)
    value = os.getenv(env_key, None)
    if value:
        path = Path(value)
    log_directory = Path('logs')
    if not log_directory.exists():
        log_directory.mkdir()
    if path.exists():
        try:
            _config = toml.load(path)
            logging.config.dictConfig(_config)
        except NameError:
            logging.basicConfig(level=default_level)
            logging.error(
                "toml module could not be imported, thus custom logging config"
                " failed. Basic logging config will be used instead."
            )
    else:
        logging.basicConfig(level=default_level)
        logging.error(
            "The specified log config file path does not exist. "
            "Basic logging config will be used instead."
        )
