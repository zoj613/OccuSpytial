from multiprocessing import cpu_count
from typing import Dict, Optional, Tuple, Union
from warnings import simplefilter

from beautifultable import BeautifulTable
from loky import get_reusable_executor
import matplotlib.pyplot as plt
import numpy as np  # type: ignore
from scipy.signal import welch as specdensity
from scipy.stats import gaussian_kde

from .icar.model import ICAR, ParamType
from .utils.utils import acf

simplefilter('ignore', UserWarning)
plt.style.use('ggplot')


class Sampler:
    """ Class doc """

    def __init__(
            self,
            X: np.ndarray,
            W: Dict[int, np.ndarray],
            y: Dict[int, np.ndarray],
            Q: np.ndarray,
            init: ParamType,
            hypers: ParamType,
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
            raise Exception("model choice can only be 'icar' or 'rsr'")
        self._names = self.model._names
        self.fullchain = np.array(self._names, ndmin=2)
        self.occ_probs = np.zeros(self.model._n)

    def _new_inits(self, init: ParamType) -> None:
        """ Function doc """
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

    def _get_samples(
            self,
            args: Tuple[
                ICAR,
                int,
                Union[int, None],
                Union[ParamType, None],
                bool,
                bool
            ]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Function doc """
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

        executor = get_reusable_executor(max_workers=cpu_count())

        if new_init is not None:
            self._new_inits(new_init)
        setattr(self, 'nonspat', nonspatial)

        args = [
            (self.model, iters, burnin, init, progressbar, nonspatial)
            for init
            in self.inits
        ]
        results = executor.map(self._get_samples, args)
        for chain, avg_occ in list(results):
            self.fullchain = np.concatenate((self.fullchain, chain))
            self.occ_probs += avg_occ

        self.occ_probs /= self.n_chains

    def trace_plots(
            self,
            show: bool = True,
            save: bool = False,
            name: str = 'traces'
    ) -> None:

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
        """ Function doc """
        if self.n_chains == 1:
            # raise Exception("the number of chains needs to be 2 or more.")
            return None
        else:
            # split up the big chain into multiple chains
            try:
                s = np.split(chains, self.n_chains, axis=0)
            except ValueError:
                raise Exception("Chains need to be of equal length.")
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
        """ Function doc """
        assert first + last <= 1, "first + last should not exceed 1"
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
