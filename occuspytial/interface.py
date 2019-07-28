from multiprocessing import Pool
import logging
from typing import Dict, Optional, Tuple, Union
from warnings import simplefilter

from beautifultable import BeautifulTable
import matplotlib.pyplot as plt
import numpy as np  # type: ignore
from scipy.signal import welch as specdensity
from scipy.stats import gaussian_kde

from .icar.model import ICAR, ParamType
from .utils.utils import acf

logger = logging.getLogger(__name__)
simplefilter('ignore', UserWarning)
plt.style.use('ggplot')
ArgsType = Tuple[
    ICAR, int, Union[int, None], Union[ParamType, None], bool, bool
]


class Sampler:
    """A class to perform all the inference for the ICAR and RSR models.

    Args:
        X (np.ndarray): Design matrix of the occupancy process.
        W (Dict[int, np.ndarray]): Design matrices of the detection
            detection process.
        y (Dict[int, np.ndarray]): Contains the detection and non-
            detection data for a species.
        Q (np.ndarray): The spatial precision matrix.
        init (ParamType): The initial values of model parameters.
        hypers (ParamType): The hyperparameters of the model.
        model (str, optional): The model name. Defaults to 'icar'.
        chains (int, optional): The number of chains to use while
            sampling. Defaults to 2.
        threshold (float, optional): The spatial autocorrelation
            parameter associated with choosing the size of the RSR
            model. Defaults to 0..

    Raises:
        Exception: If unsupported model name is used as input.

    Attributes:
        mode (str): The model name.
        n_chains (int): The number of chains to use when sampling.
        inits (list): Parameter initial values for each chain.
        fullchain (np.ndarray): The combined posterior samples from the
            different chains.
        occ_probs (np.ndarray): Estimated occupancy probabilities for
            each site, using the combined samples.

    Methods:
        run: Performs the posterior sampling of parameters of interest,
        trace_plots: Gets the trace plots of the posterior samples.
        corr_plots: Gets the autocorrelation plots of the samples.
        gelman: Performs the Gelman-Rubin test on the posterior samples.
        geweke: Performs the Geweke test on the posterior samples.
    """

    def __init__(
            self,
            X: np.ndarray,
            W: Dict[int, np.ndarray],
            y: Dict[int, np.ndarray],
            Q: np.ndarray,
            init: Optional[ParamType] = {},
            hypers: Optional[ParamType] = {},
            model: str = 'icar',
            chains: int = 2,
            threshold: float = 0.
    ) -> None:

        self.mode = model
        self.n_chains = chains
        self.inits = [init]
        self._new_inits(init)  # set initial values for the additional chains
        if model.lower() == 'icar':
            self.model = ICAR(X, W, y, Q, init, hypers)
        elif model.lower() == 'rsr':
            self.model = ICAR(
                X, W, y, Q, init, hypers, use_rsr=True, threshold=threshold
            )
        else:
            logger.error(f"wrong model choice. {model} is not supported.")
            raise Exception("model choice can only be 'icar' or 'rsr'")
        self._names = self.model._names
        self.fullchain = np.array(self._names, ndmin=2)
        self.occ_probs = np.zeros(self.model._n)

    def _new_inits(self, init: ParamType) -> None:
        """ Set new initial parameter values. """
        self.inits = [init]
        for _ in range(1, self.n_chains):
            # create multiple initial values for the additional chains
            # using random pertubation of the user supplied initial values
            _init = {}
            for key, value in init.items():
                if key == "alpha" or key == "beta":
                    value += np.random.uniform(-2, 2, len(value))
                elif key == "tau":
                    value += np.random.uniform(0, 5)
                else:
                    value += np.random.uniform(-2, 2, len(value))

                _init[key] = value
            self.inits.append(_init)

    def __call__(self, args: ArgsType) -> Tuple[np.ndarray, np.ndarray]:
        model, iters, burnin, init, progressbar, nonspat = args
        model.run_sampler(iters, burnin, init, progressbar, nonspat)
        return model._traces, model.z_mat.mean(axis=0)

    def run(
            self,
            iters: int = 1000,
            burnin: Optional[int] = None,
            new_init: Optional[ParamType] = None,
            progressbar: bool = True,
            nonspatial: bool = False
    ) -> None:
        """Perform the sampling of posterior parameters of the model.
        Sampling ia done in parallel for each of the number of chains
        specified in the 'n_chains' attribute.

        Args:
            iters (int, optional): The number of sampler iterations.
                Defaults to 1000.
            burnin (Optional[int]): The size of the burnin samples to be
                thrown away. Defaults to None.
            new_init (Optional[ParamType]): Parameter initial values to
                use for sampling. If not supplied then the values used
                are be the ones the class instance was initialized with.
                Defaults to None.
            progressbar (bool, optional): Whether or not to display the
                progress of the sampling process in the terminal.
                Defaults to True.
            nonspatial (bool, optional): Whether or not to sample from
                the model version without the spatial part included.
                Defaults to False.
        """
        if new_init is not None:
            logger.debug("setting parameter initial values")
            self._new_inits(new_init)
        setattr(self, 'nonspat', nonspatial)

        args = [
            (self.model, iters, burnin, init, progressbar, nonspatial)
            for init
            in self.inits
        ]
        logger.debug("distributing the sampler among workers...")
        with Pool() as pool:
            logger.debug(f"Pool currently using {pool._processes} processes.")
            for chain, avg_occ in pool.map(self, args):
                self.fullchain = np.concatenate((self.fullchain, chain))
                self.occ_probs += avg_occ
        logger.info("sampling completed successfully.")
        self.occ_probs /= self.n_chains

    def trace_plots(
            self,
            show: bool = True,
            save: bool = False,
            name: str = 'traces'
    ) -> None:
        """Generate the traceplots of the posterior samples of the para-
        meters of interest. The plots can be displayed in one single
        figure on the screen and/or saved as a picture. A custom name
        can be given to the saved file.

        Args:
            show (bool, optional): Whether to show the plot on screen.
                Defaults to True.
            save (bool, optional): Whether to save plot as a picture.
                Defaults to False.
            name (str, optional): Name of the picture if save = True.
                The supplied input can also be a path if it is to be
                saved in a custom directory. Defaults to 'traces'.
        """
        traces = self.fullchain
        if self.nonspat:
            plot_rows = traces.shape[1] - 1
        else:
            plot_rows = traces.shape[1]
        plt.figure(figsize=(6, 8))
        for i in range(plot_rows):
            data = traces[:, i][1:].astype(np.float64)
            plt.subplot(plot_rows, 2, 2*i + 1)
            plt.plot(data)
            plt.title(self._names[i])
            plt.subplot(plot_rows, 2, 2*i + 2)

            s_data = sorted(data)
            plt.plot(s_data, gaussian_kde(data).pdf(s_data), linewidth=2)
            plt.hist(
                data,
                bins=55,
                density=True,
                histtype='stepfilled',
                color='red',
                alpha=0.3
            )
            plt.ylabel('')
        plt.tight_layout()
        if save:
            plt.savefig(name, format='svg', bbox_inches='tight')
        plt.show() if show else plt.clf()

    def corr_plots(
            self,
            num_lags: int = 50,
            show: bool = True,
            save: bool = False,
            name: str = 'corr'
    ) -> None:
        """Generate the autocorrelation plots of the posterior samples
        of the parameters of interest. The plots can be displayed in one
        single figure on the screen and/or saved as a picture. A custom
        name/path can be given for the saved file.

        Args:
            num_lags (int, optional): The maximum number of lags to dis-
                play in the plot. Defaults to 50.
            show (bool, optional): Whether to show the plot on screen.
                Defaults to True.
            save (bool, optional): Whether to save plot as a picture.
                Defaults to False.
            name (str, optional): Name of the picture if save = True.
                The supplied input can also be a path if it is to be
                saved in a custom directory. Defaults to 'traces'.
        """
        traces = self.fullchain
        if self.nonspat:
            plot_rows = traces.shape[1] - 1
        else:
            plot_rows = traces.shape[1]
        plt.figure(figsize=(6, 8))
        for i in range(plot_rows):
            data = traces[:, i][1:].astype(np.float64)
            lagdata = [acf(data, lag=i) for i in range(0, num_lags + 1)]
            plt.subplot(
                plot_rows,
                1, i + 1,
                xlim=(-1, num_lags + 1),
                ylim=(min(lagdata) - 0.2 if min(lagdata) < 0 else -0.2, 1.2)
            )
            plt.plot(lagdata, 'C0o', markersize=5)
            ymaxs = [y - 0.05 if y > 0 else y + 0.05 for y in lagdata]
            plt.vlines(np.arange(num_lags + 1), ymin=0, ymax=ymaxs, color='k')
            plt.hlines(y=0, xmin=-1, xmax=num_lags + 1, color='C0')
            plt.title(f"acf of {self._names[i]}")
        plt.tight_layout()
        if save:
            plt.savefig(name, format='svg', bbox_inches='tight')
        plt.show() if show else plt.clf()

    def gelman(self, chains: np.ndarray) -> Union[None, np.ndarray]:
        """Perform the Gelman-Rubin convergence diagnostics test.

        The test is performed using the posterior sample chains that are
        generated. A numpy 1D-array is returned with each element
        corresponding to a posterior parameter in the same order that
        they appear in the fullchains attribute's first row.

        Args:
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

    def geweke(
            self,
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
        gewe = self.geweke(fullchain)
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
            rhat = self.gelman(fullchain)

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
