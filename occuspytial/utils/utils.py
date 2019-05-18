from datetime import timedelta
import sys
import time
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore
from numpy.random import standard_normal as std_norm
from scipy.linalg import cholesky as chol
from scipy.sparse import csc_matrix, issparse
try:
    from sksparse.cholmod import cholesky as sp_chol
except ImportError:
    pass


class ProgressBar:
    """ Class doc """
    BAR_LENGTH = 25

    def __init__(self, n: int) -> None:
        """ Class initialiser """
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

        now = self._now
        elapsed_time = timedelta(seconds=now - self.start)
        est_total_time = timedelta(
            seconds=self.n / self.i * (now - self.start)
        )
        return elapsed_time, est_total_time - elapsed_time

    def update(self) -> None:

        self.i += 1
        if sys.stdout.isatty():
            print(self._bar_string(), file=self._FILE, end='\r')
        else:
            print(self._bar_string(), end='\r')

        if self.progress == 1:
            print()


def affine_sample(
        mean: np.ndarray,
        cov: Union[csc_matrix, np.ndarray],
        return_factor: bool = False
) -> Union[Tuple[np.ndarray, sp_chol], np.ndarray]:
    """ Function doc """
    try:
        factor = sp_chol(cov, mode="supernodal")
        chol_factor = factor.apply_Pt(factor.L())
        x = mean + chol_factor @ std_norm(mean.size)

    except (NameError, AttributeError):

        # sp_chol failed to import and/or cov is dense
        cov_dense = cov.toarray() if issparse(cov) else cov
        factor = chol(cov_dense, check_finite=False).T
        x = mean + factor @ std_norm(mean.size)

    finally:

        if return_factor:
            return x, factor
        else:
            return x


def acf(x: np.ndarray, lag: int = 0) -> float:
    """ Function doc """
    if lag == 0:
        return 1
    elif lag < len(x) - 1:
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    else:
        raise Exception("lag must be less than {}".format(len(x) - 1))


class CustomDict(dict):
    # a mix-in class for group-indexing W and y dictionaries
    def slice(self, *keys: Sequence[Any]) -> np.ndarray:
        try:
            out = [self[k] for k in keys]
        except TypeError:  # if input is not hashable
            out = [self[k] for k in keys[0]]
        return np.concatenate(tuple(out))


class SpatialStructure:
    """ A class intended for generating spatial precision matrix used in
    models like CAR, ICAR and RSR"""

    def __init__(self, n: int) -> None:
        """ Class initialiser """
        self.n = n

    def _generate_random_lattice(
            self,
            n: Optional[int] = None,
            fix_square: bool = False
    ) -> None:
        """ Function doc """
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
        """ Function doc """
        assert n_type in [4, 8], "n_type must be 4 or 8"
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

        for item in np.array(out):
            if not np.all(item >= 0):
                out.remove(tuple(item))
        return out

    def _adjacency_matrix(self, n_type: int = 48) -> None:
        """ use the generated lattice to create an adjacency matrix A, where
        an element A[i, j] = 1 if i and j are neighbors and A[i, j] = 0
        otherwise for i =/= j. A[i, i] = 0 for all i."""
        a = np.zeros((self.n, self.n))
        for indx, site in np.ndenumerate(self.lattice):
            a[site - 1, site - 1] = 0
            # randomly decide the maximum number of neighbors for site.
            if n_type == 'mixed':
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
            rho: int = 1,
            square_lattice: bool = False
    ) -> np.ndarrays:
        """ Function doc """
        self._generate_random_lattice(fix_square=square_lattice)
        self._adjacency_matrix(n_type)
        d_mat = np.diag(self.A.sum(axis=0))
        return d_mat - rho * self.A
