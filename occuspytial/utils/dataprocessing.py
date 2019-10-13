import logging
from operator import itemgetter
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


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
        return np.array(visits, dtype=np.uint64)


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
