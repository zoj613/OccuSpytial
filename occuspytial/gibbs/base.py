import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc
from scipy.sparse.linalg import eigsh
from tqdm.auto import tqdm

from ..chain import Chain
from ..data import Data
from ..posterior import PosteriorParameter
from ..utils import get_generator

from .parallel import sample_parallel
from .state import State, FixedState


class _GibbsState(State):
    """Posterior parameter state container.

    This class is used to conveniently retrieve the posterior samples of the
    parameters of interest ('alpha', 'beta', 'tau') after each Gibbs sampler
    iteration.
    """

    _posterior_names = ('alpha', 'beta', 'tau')

    @property
    def posteriors(self):
        return {key: self.__dict__[key] for key in self._posterior_names}


class GibbsBase:
    """Base class for Gibbs samplers of species spatial occupancy models.

    Parameters
    ----------
    Q : np.ndarray
        Spatial precision matrix of spatial random effects.
    W : Dict[int, np.ndarray]
        Dictionary of detection corariates where the keys are the site numbers
        of the surveyed sites and the values are arrays containing
        the design matrix of each corresponding site.
    X : np.ndarray
        Design matrix of species occupancy covariates.
    y : Dict[int, np.ndarray]
        Dictionary of survey data where the keys are the site numbers of the
        surveyed sites and the values are number arrays of 1's and 0's
        where 0's indicate "no detection" and 1's indicate "detection". The
        length of each array equals the number of visits in the corresponding
        site.
    hparams : {None, Dict[str, Union[float, np.ndarray]}, optional
        Hyperparameters of the occupancy model. valid keys for the dictionary
        are:
            - ``a_mu``: mean of the normal prior of detection covariates.
            - ``a_prec``: precision matrix of the normal prior of detection
              covariates.
            - ``b_mu``: mean of the normal prior of occupancy covariates.
            - ``b_prec``: precision matrix of the normal prior of occupancy
              covariates.
            - ``tau_rate``: rate parameter of the Gamma prior of the spatial
              parameter.
            - ``tau_shape``: shape parameter of the Gamma prior of the spatial
              parameter.
    random_state : {None, int, numpy.random.SeedSequence}
        A seed to initialize the bitgenerator.

    Attributes
    ----------
    chain : Chain
        Posterior chain object. An instance of :class:`occuspytial.chain.Chain`
    dists : :class:`~occuspytial.gibbs.state.FixedState`
        Container for conditional probability distribution instances of
        posterior parameters.
    fixed : :class:`~occuspytial.gibbs.state.FixedState`
        A Container for fixed values and parameters who's values remain
        constant during sampling. This variable must be an instance of
        :class:`~occuspytial.gibbs.state.FixedState`.
    rng : numpy.random.Generator
        Instance of numpy's Generator class, which exposes a number of random
        number generating methods.
    state : :class:`~occuspytial.gibbs.state.State`
        A container to store model parameter values and other variables whose
        values change at various stages during sampling.
    """

    def __init__(self, Q, W, X, y, hparams=None, random_state=None):
        self.W = Data(W)
        self.X = X
        self.y = Data(y)
        self.rng = get_generator(random_state)

    def step(self):
        """Step through one iteration of the sampler.

        This method should execute the parameter update steps the Gibbs sampler
        should take in order to complete one iteration of the sampler.

        Raises
        ------
        NotImplementedError
            If `sample` method is called but subclass does not implement a
            `step` method.

        """
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement a `step` method.'
        )

    def _configure(self, Q, hparams, verify_precision=True, **kwargs):
        """To be called by subclasses in order to configure the sampler."""
        if verify_precision:
            self._verify_spatial_precision(Q)

        self.state = _GibbsState()
        self.state.z = np.ones(self.X.shape[0])
        surveyed = self.y.surveyed

        # sites where a species is observed on any visit are given an
        # occupany state of 1 since observation implies occupancy.
        self.state.z[surveyed] = [any(self.y[site]) for site in surveyed]
        self.state.k = self.state.z - 0.5

        self.fixed = FixedState()
        self.fixed.Q = Q if isspmatrix_csc(Q) else csc_matrix(Q)
        self.fixed.n = self.X.shape[0]
        self.fixed.ones = np.ones(self.fixed.n)
        self.fixed.not_surveyed = [
            site for site in range(self.fixed.n) if site not in surveyed
        ]

        # get site numbers of surveyed sites where the species was not observed
        # on any visit.
        self.fixed.not_obs = [i for i in surveyed if not self.state.z[i]]
        self.fixed.obs = [i for i in surveyed if self.state.z[i]]

        # n_no == number of surveyed sites where species was not observed.
        self.fixed.n_no = len(self.fixed.not_obs)
        # n_ns == number of sites not surveyed out of n sites.
        self.fixed.n_ns = len(self.fixed.not_surveyed)

        # row-wise stacked design matrices of conditional detection covariates.
        # the matrices are of surveyed sites where a species was not observed.
        self.fixed.W_not_obs = self.W[self.fixed.not_obs]

        # Number of visits per surveyed site where species was not observed.
        self.fixed.visits_not_obs = self.W.visits(self.fixed.not_obs)

        # `stacked_w_indices` contains the index range of each site where the
        # species was not observed during survey. The range size is equal to
        # the number of visits. This array is convenient for updating ``z``
        # parameter of the occupancy model during sampling when the link
        # function used to specify the model is a logit function.
        sections = np.cumsum(self.fixed.visits_not_obs)
        self.fixed.stacked_w_indices = np.pad(sections, (1, 0))[:-1]

        if hparams:
            self.fixed = self._set_hyperparams(self.fixed, hparams)
        else:
            self.fixed = self._set_default_hyperparams(self.fixed)

        # value of the hyperparameter's (alpha and beta) precision matrix
        # multiplied by its mean
        self.fixed.a_prec_by_mu = self.fixed.a_prec @ self.fixed.a_mu
        self.fixed.b_prec_by_mu = self.fixed.b_prec @ self.fixed.b_mu

        self.dists = FixedState()

    def _verify_spatial_precision(self, Q):
        """Check if Q is not singular by computing smallest eigenvalue."""
        eig = eigsh(Q, k=1, which='SA', return_eigenvectors=False, sigma=0.001)
        if eig[0] >= 1e-4:
            raise ValueError('Spatial precision matrix Q must be singular.')

    def _set_hyperparams(self, params, hyperparams):
        for key, value in hyperparams.items():
            setattr(params, key, value)
        return params

    def _set_default_hyperparams(self, params):
        params.tau_rate = 0.005
        params.tau_shape = 0.5 + 0.5 * (self.fixed.n - 1)
        alpha_size = self.W[self.W.surveyed[0]].shape[1]
        params.a_mu = np.zeros(alpha_size)
        params.a_prec = np.eye(alpha_size) / 10
        beta_size = self.X.shape[1]
        params.b_mu = np.zeros(beta_size)
        params.b_prec = np.eye(beta_size) / 10
        return params

    def _initialize_posterior_state(self, start=None):
        """Set sampler starting values."""
        if start is None:
            self.state = self._initialize_default_start(self.state)
        else:
            self.state.alpha = start['alpha']
            self.state.beta = start['beta']
            self.state.tau = start['tau']
            self.state.eta = start['eta']
            self.state.spatial = self.state.eta

    def _initialize_default_start(self, state):
        """Set starting values for the gibbs sampler when not given."""
        state.tau = self.rng.gamma(0.5, 1 / self.fixed.tau_rate)
        eta = self.rng.standard_normal(self.fixed.n)
        eta = eta - eta.mean()
        state.eta = eta
        state.spatial = self.state.eta
        state.alpha = self.rng.multivariate_normal(
            self.fixed.a_mu, 100 * self.fixed.a_prec, method='cholesky'
        )
        state.beta = self.rng.multivariate_normal(
            self.fixed.b_mu, 100 * self.fixed.b_prec, method='cholesky'
        )
        return state

    def _run(
        self, size, burnin=0, start=None, chains=2, progressbar=True, pos=0
    ):
        """Contains the sampler's logic for obtaining posterior samples.

        It is not meant to be called explicitely. It is only meant to be called
        by the ``sample`` method or by a parallel sampling method when using
        multiple chains. The `pos` parameter indicates the position of the
        progress bar on the console. It is meant to be used by the function
        :func:`~occuspytial.gibbs.parallel.sample_parallel` to correctly
        position the progress bar of each sampler chain.
        """
        self._initialize_posterior_state(start)
        chain_params = {
            'alpha': self.state.alpha.size,
            'beta': self.state.beta.size,
            'tau': 1
        }
        self.chain = Chain(chain_params, size - burnin)
        tqdm_iterator = tqdm(
            range(size), total=size, disable=not progressbar, position=pos
        )
        for i in tqdm_iterator:
            self.step()
            if i >= burnin:
                self.chain.append(self.state.posteriors)

        return self.chain

    def sample(self, size, burnin=0, start=None, chains=2, progressbar=True):
        r"""Obtain posterior samples of the parameters of interest.

        Only parameters {``alpha``, ``beta``, ``tau``} are stored after
        sampling, meaning the conditional detection and occupancy covariate
        coefficients, plus the posterior spatial parameter :math:`\\tau` .

        Parameters
        ----------
        size : int
            The number of total samples to generate.
        burnin : int, optional
            The number of initial samples to discard as "burnin" samples.
            `burnin` must be less than `size`.
        start : {None, Dict[str, np.ndarray]}, optional
            Starting values of the parameters ``alpha``, ``beta``, ``tau`` and
            ``eta``.
        chains : int, optional
            Number of chains to generate in parallel. Defauls to 1.
        progressbar : bool, optional
            Whether to display the progress bar during sampling. Defaults to
            True.

        Returns
        -------
        out : :class:`~occuspytial.posterior.PosteriorParameter`
            Posterior samples of the parameters of interest.

        Raises
        ------
        ValueError
            If the burnin values is larger than the number of samples requested
            or when the number of chains is not a positive integer.

        """
        if burnin >= size:
            raise ValueError('burnin value cannot be larger than sample size')
        if chains < 1:
            raise ValueError('chains must a positive integer.')

        samples = sample_parallel(
            self,
            size=size,
            burnin=burnin,
            chains=chains,
            start=start,
            progressbar=progressbar
        )
        return PosteriorParameter(*samples)

    def copy(self):
        """Create a copy of this classes's instance.

        Returns
        -------
        out : self
            A copy of this object's instance with the same attribute values.
        """
        out = type(self).__new__(self.__class__)
        out.__dict__.update(self.__dict__)
        # make sure the copy has its own unique random number generator
        seed_seq = self.rng._bit_generator._seed_seq.spawn(1)[0]
        out.__dict__['rng'] = get_generator(seed_seq)
        return out
