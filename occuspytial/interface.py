from multiprocessing import Pool
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np  # type: ignore

from occuspytial.icar.model import ICAR, ParamType
from occuspytial.utils.stats import ConvergenceDiagnostics
from occuspytial.utils.visualization import Plots

logger = logging.getLogger(__name__)
ArgsType = Tuple[
    ICAR, int, Union[int, None], Union[ParamType, None], bool, bool
]


class Sampler(ConvergenceDiagnostics, Plots):
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
            init: Optional[ParamType] = None,
            hypers: Optional[ParamType] = None,
            model: str = 'icar',
            chains: int = 2,
            threshold: float = 0.
    ) -> None:

        self.mode = model
        self.n_chains = chains
        if model.lower() == 'icar':
            self.model = ICAR(X, W, y, Q, init, hypers)
        elif model.lower() == 'rsr':
            self.model = ICAR(
                X, W, y, Q, init, hypers, use_rsr=True, threshold=threshold
            )
        else:
            logger.error(f"wrong model choice. {model} is not supported.")
            raise Exception("model choice can only be 'icar' or 'rsr'")
        # set initial values for the additional chains
        self._new_inits(self.model.init.__dict__)
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
                    new_value = value + np.random.uniform(-2, 2, len(value))
                elif key == "tau":
                    new_value = value + np.random.uniform(0, 5)
                else:
                    new_value = value + np.random.uniform(-2, 2, len(value))

                _init[key] = new_value
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
            nonspatial: bool = False,
            regularize: Optional[float] = None
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

        if regularize is not None and self.model.__class__.__name__ == 'ICAR':
            self.model.regularize = regularize

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
