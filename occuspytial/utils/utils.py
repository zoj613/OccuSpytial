from datetime import timedelta
import sys
import time
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore
from numpy.random import standard_normal as std_norm
from scipy.linalg import cholesky as chol
from scipy.sparse import csc_matrix, issparse
try:
    from sksparse.cholmod import Factor, cholesky as sp_chol
except ImportError:
    pass


class ProgressBar:
    """A class to emulate a progress bar on the terminal

    Attributes:
        BAR_LENGTH (int): The horizontal length of the progress bar.
        _FILE (io.TextIOWrapper): A file object.
        start (float): The start time, created using time.monotonic()
        n (int): Total number of iterations.
        i (int): The current iteration.
        fill (str): The progress bar fill. Either a solid bar or hash.
        progress (float): A value between 0 and 1 (inclusive) indicating
            the percentage progress.

    Methods:
        update: Updates the progress forward by one iteration.
    """
    BAR_LENGTH = 25

    def __init__(self, n: int) -> None:
        """
        Args:
            n (int): Total number of iterations.
        """
        if sys.stdout.isatty():  # check if script is running from console
            self._FILE = open(sys.stdout.fileno(), mode='w', encoding='utf8')
            fill = "█"
        else:
            fill = "#"

        self.start = self._now
        self.n = n
        self.i = 0
        self.fill = fill
        self.progress = 0

    def _bar_string(self) -> str:
        """Describes the progress bar output string"""
        elapsed, remaining = self._elapsed_and_remaining_time()
        self.progress = self.i / self.n
        block = int(round(ProgressBar.BAR_LENGTH * self.progress))
        bar = [
            self.fill * block + ' ' * (ProgressBar.BAR_LENGTH - block),
            self.progress * 100,
            str(elapsed).split('.')[0],
            self.i,
            self.n,
            str(remaining).split(".")[0],
            round(self.i / elapsed.total_seconds(), 2)
        ]
        return '{1:.1f}%[{0}] {3}/{4} [{2}<{5}, {6}draws/s]'.format(*bar)

    @property
    def _now(self) -> float:

        return time.monotonic()

    def _elapsed_and_remaining_time(self) -> Tuple[timedelta, timedelta]:
        """Calculates the elapsed time and remaining time"""
        now = self._now
        elapsed_time = timedelta(seconds=now - self.start)
        est_total_time = timedelta(
            seconds=self.n / self.i * (now - self.start)
        )
        return elapsed_time, est_total_time - elapsed_time

    def update(self) -> None:
        """Update the progress bar output"""
        self.i += 1
        # decide how to print the bar depending on whether standard
        # output is the terminal or not.
        if sys.stdout.isatty():
            print(self._bar_string(), file=self._FILE, end='\r')
        else:
            print(self._bar_string(), end='\r')
        # stoping condition, finishing printing if progress is at 100%.
        if self.progress == 1:
            print()


def affine_sample(
        mean: np.ndarray,
        cov: Union[csc_matrix, np.ndarray],
        return_factor: bool = False
) -> Union[Tuple[np.ndarray, Factor], np.ndarray]:
    """Sample from a multivariate normal distribution that has a sparse
    covariance matrix using Affine transformation / "reparameterization
    trick".

    Args:
        mean (np.ndarray): A mean vector.
        cov (Union[csc_matrix, np.ndarray]): Covariance matrix
        return_factor (bool, optional): Whether or not the function
        . Defaults to False.

    Returns:
        Union[Tuple[np.ndarray, Factor], np.ndarray]: A random sample
            from the multivariate normal distribution together with the
            Cholesky factor of the covariance matrix if return_factor is
            set to True. The Cholesky factor is stored efficiently as
            a sksparse.cholmod.Factor object.
    """
    try:
        factor = sp_chol(cov, mode="supernodal")
        chol_factor = factor.apply_Pt(factor.L())
        x = mean + chol_factor @ std_norm(mean.size)
    # sp_chol failed to import and/or cov is dense:
    except (NameError, AttributeError):

        # convert the covariance into a dense matrix if it stored in
        # a sparse format.
        cov_dense = cov.toarray() if issparse(cov) else cov
        factor = chol(cov_dense, check_finite=False).T
        x = mean + factor @ std_norm(mean.size)

    finally:

        if return_factor:
            return x, factor
        else:
            return x


def acf(x: np.ndarray, lag: int = 0) -> float:
    """Calculate the autocorrelation of a series of values using a
    specified lag size.

    Args:
        x (np.ndarray): The series of values.
        lag (int, optional): The size of the lag. Defaults to 0.

    Raises:
        Exception: If the lag is equal to or larger than the length of
             the series.

    Returns:
        float: The sample autocorrelation of x.
    """
    lag = abs(lag)  # ensure function works with negative lag values.
    if lag == 0:
        return 1
    elif lag < len(x) - 1:
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    else:
        raise Exception(f"lag must be less than {len(x) - 1}")


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
    models like CAR, ICAR and RSR

    Attributes:
        n (int): The number of sites.

    Methods:
        spatial_precision: Returns a randomly generated spatial presicion
            matrix of size n by n.
    """

    def __init__(self, n: int) -> None:
        """
        Args:
            n (int): The number of sites.
        """
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
            or indx[0] == (self.n - 1)  # is in last row of lattice
            or indx[1] == (self.n - 1)  # is in last column of lattice
        )
        if edge_case:
            for item in np.array(out):
                if np.any(item < 0) or np.any(item >= self.n):
                    out.remove(tuple(item))
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
                try:
                    neighbor_site = self.lattice[row, col]
                    a[site - 1, neighbor_site - 1] = 1
                    a[neighbor_site - 1, site - 1] = 1
                except IndexError:
                    continue
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
