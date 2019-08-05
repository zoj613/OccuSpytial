import logging
from typing import Tuple, Union
import warnings

from beautifultable import BeautifulTable
import numpy as np
from scipy.signal import welch as specdensity
from numpy.random import standard_normal as std_norm
from scipy.linalg import cholesky as chol
from scipy.sparse import csc_matrix, issparse

try:
    from sksparse.cholmod import Factor, cholesky as sp_chol
    FactorObject = Factor
except ImportError:
    warnings.showwarning(
        "scikit sparse is not installed. Inference may be slow. "
        "To ensure maximum speed during inference, please install the "
        "sckit-sparse package.",
        category=ImportWarning,
        filename=__name__,
        lineno=14
    )
    FactorObject = None

warnings.simplefilter('ignore', UserWarning)
logger = logging.getLogger(__name__)


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


def affine_sample(
        mean: np.ndarray,
        cov: Union[csc_matrix, np.ndarray],
        return_factor: bool = False
) -> Union[Tuple[np.ndarray, FactorObject], np.ndarray]:
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
    if issparse(cov) and 'sp_chol' in globals().keys():
        factor = sp_chol(cov, ordering_method="metis")
        chol_factor = factor.apply_Pt(factor.L())
        x = mean + chol_factor @ std_norm(mean.size)
    else:
        cov_dense = cov.toarray() if issparse(cov) else cov
        factor = chol(cov_dense, check_finite=False)
        x = mean + std_norm(mean.size) @ factor

    if return_factor:
        return x, factor
    else:
        return x


class ConvergenceDiagnostics:

    def gelman(self, chains: np.ndarray) -> Union[None, np.ndarray]:
        """Perform the Gelman-Rubin convergence diagnostics test.

        The test is performed using the posterior sample chains that are
        generated. A numpy 1D-array is returned with each element
        corresponding to a posterior parameter in the same order that
        they appear in the fullchains attribute's first row.

        Args:
            self (object): A sampler class containing the attribute `n_chains`
            chains (np.ndarray): The combined parameter samples.

        Raises:
            ValueError: If chains are not of equal length

        Returns:
            Union[None, np.ndarray]: If the number of chains is 1 then
            return None, else return test values for each parameter.
        """
        if self.n_chains == 1:
            return None
        else:
            # split up the big chain into multiple chains
            try:
                s = np.split(chains, self.n_chains, axis=0)
            except ValueError as ve:
                logger.exception("Unequal chain lengths.")
                raise Exception("Chains need to be of equal length.") from ve
            m = self.n_chains  # number of chains
            n = s[0].shape[0]  # length of each chain
            # squared differences of each chain
            sqd = [
                (s[i].mean(axis=0) - chains.mean(axis=0)) ** 2
                for i in range(m)
            ]
            # calculate the between chains variances for each parameter
            b = np.stack(sqd).sum(axis=0) * s[0].shape[0] / (len(s) - 1)

            # sample variance of each chain
            sv = [np.var(s[i], axis=0, ddof=1) for i in range(m)]
            # within-chain variances  for each parameter
            w = np.stack(sv).mean(axis=0)

            # the pooled variance
            v = (n - 1) * w / n + (m + 1) * b / (m * n)

            return np.sqrt(v / w)

    @classmethod
    def geweke(
            cls,
            chain: np.ndarray,
            first: float = 0.1,
            last: float = 0.5
    ) -> np.ndarray:
        """Performs the Geweke convergence diagnostics test.

        In this implementation the Spectral density used to estimate the
        variance of the sample is estimated using SciPy's welch method
        from the signal submodule.

        Args:
            chain (np.ndarray): The posterior samples of the parameters.
            first (float, optional): The first portion of the samples.
                Defaults to 0.1.
            last (float, optional): The last portion of the samples.
                Defaults to 0.5.

        Returns:
            np.ndarray: Test values for each parameter.
        """
        if first + last > 1:
            _msg = "first + last should not exceed 1"
            logger.error(_msg)
            raise ValueError(_msg)
        x1 = chain[:int(first * chain.shape[0])]
        x2 = chain[int((1 - last) * chain.shape[0]):]
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        x1mean = x1.mean(axis=0)
        x2mean = x2.mean(axis=0)

        num_of_params = chain.shape[1]
        s1 = np.empty(num_of_params)
        s2 = s1

        for i in range(num_of_params):
            s1[i] = specdensity(x1[:, i])[1][0]
            s2[i] = specdensity(x2[:, i])[1][0]

        return (x1mean - x2mean) / np.sqrt(s1 / n1 + s2 / n2)

    @property
    def summary(self) -> BeautifulTable:
        """A summary table of the posterior sampling results.

        The table can be indexed using an int, str or slice object. See
        beatifultable documentation for more information on how to index
        a BeautifulTable object.

        Args:
            self (object): A class with attributes fullchain, n_chains,
                _names, nonspat and n_chains.

        An example table:
          param      mean      std       2.5%     97.5%   PSRF  geweke
        alpha_0    -0.101    0.236     -0.563      0.36  1.011   4.958
        alpha_1     1.689    0.299      1.103     2.275  1.002  -0.169
         beta_0     0.216    0.309      -0.39     0.822  1.001   5.408
         beta_1    -0.382    0.328     -1.026     0.261  0.999   -1.39
         beta_2     -0.08    0.314     -0.696     0.536  1.004  -1.663
            PAO     0.547    0.052      0.445      0.65    1.0    4.84
            tau  1013.798  1453.14  -1834.357  3861.952  1.001   2.112
        """
        table = BeautifulTable(default_alignment=BeautifulTable.ALIGN_RIGHT)
        table.set_style(BeautifulTable.STYLE_NONE)
        fullchain = self.fullchain[1:].astype(np.float64)
        gewe = ConvergenceDiagnostics.geweke(fullchain)
        means = fullchain.mean(axis=0)
        stds = fullchain.std(axis=0, ddof=1)
        lower = means - 1.96 * stds
        upper = means + 1.96 * stds

        if self.n_chains == 1:
            table.column_headers = [
                'param', 'mean', 'std', '2.5%', '97.5%', 'geweke'
            ]
        else:
            table.column_headers = [
                'param', 'mean', 'std', '2.5%', '97.5%', 'PSRF', 'geweke'
            ]
            rhat = ConvergenceDiagnostics.gelman(self, fullchain)

        names = self._names[:-1] if self.nonspat else self._names

        for i, param in enumerate(names):
            param = param.replace('$', '')
            param = param.replace('\\', '')
            if self.n_chains == 1:
                table.append_row(
                    [param] + [means[i], stds[i], lower[i], upper[i], gewe[i]]
                )
            else:
                table.append_row(
                    [param] +
                    [means[i], stds[i], lower[i], upper[i], rhat[i], gewe[i]]
                )

        return table
