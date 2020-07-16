import warnings

import numpy as np


def get_generator(random_state):
    """Get an instance of a numpy random number generator object.

    This instance uses the SFC64 bit generator, which is the fastest numpy
    currently has to offer as of version 1.19. This function conveniently
    instantiates a generator of this kind and should be used in all modules.

    Parameters
    ----------
    random_state : {None, int, array_like[ints], numpy.random.SeedSequence}
        A seed to initialize the bitgenerator.

    Returns
    -------
    numpy.random.Generator
        Instance of numpy's Generator class, which exposes a number of random
        number generating methods.
    """
    bitgenerator = np.random.SFC64(random_state)
    return np.random.default_rng(bitgenerator)


def rand_precision_mat(lat_row, lat_col, max_neighbors=8, rho=1):
    """Generate a random spatial precision matrix.

    The spatial precision matrix is generated using a rectengular lattice
    of dimensions `lat_row` x `lat_col`, and thus the row and colum size of
    the matrix is (`lat_row` x `lat_col`).

    Parameters
    ----------
    lat_row : int
        Number of rows of the lattice used to generate the matrix.
    lat_col : int
        Number of columns of the lattice used to generate the matrix.
    max_neighbors : {4, 8}, optional
        The maximum number of neighbors for each site. The default is 8.
    rho : float, optional
        The spatial weight parameter. Takes values between 0 and 1, with
        0 implying independent random effects and 1 implying strong spatial
        autocorrelation. Setting the value to 1 is equivalent to generating
        the Intrinsic Autoregressive Model.

    Returns
    -------
    scipy.sparse.coo_matrix
        Spatial precision matrix

    Raises
    ------
    ValueError
        If the `max_neighbours` is any value other than 4 or 8.
    """
    if max_neighbors == 8:
        nn = 'queen'
    elif max_neighbors == 4:
        nn = 'rook'
    else:
        raise ValueError('Maximum number of neighbors should be one of {4, 8}')

    with warnings.catch_warnings():
        # ignore the "geopandas not available" warning since it is not relevant
        warnings.simplefilter('ignore', UserWarning)
        import libpysal

    W = libpysal.weights.lat2SW(lat_row, lat_col, criterion=nn, row_st=False)
    W = W.tocoo()
    D = W.sum(axis=1).A1
    W.data = -W.data * rho
    W.setdiag(D)
    return W


def make_data(
    n=150,
    min_v=None,
    max_v=None,
    ns=None,
    p=3,
    q=3,
    tau_range=(0.25, 1.5),
    max_neighbors=8,
    random_state=None,
 ):
    """Generate random data to use for modelling species occupancy.

    Parameters
    ----------
    n : int, optional
        Number of sites. Defaults to 150.
    min_v : int, optional
        Minimum number of visits per site. If None, the maximum number is set
        to 2. Defaults to None.
    max_v : int, optional
        Maximum number of visits per site. If None, the maximum number is set
        to 10% of `n`. Defaults to None.
    ns : int, optional
        Number of surveyed sites out of `n`. If None, then this parameter is
        set to 50% of `n`. Defaults to None.
    p : int, optional
        Number covariates to use for species occupancy. Defaults to 3.
    q : int, optional
        Number of covariates to use for conditonal detection. Defaults to 3.
    tau_range : tuple, optional
        The range to randomly sample the precision parameter value from.
        Defaults to (0.25, 1.5).
    max_neighbors : int, optional
        Maximum number of neighbors per site. Should be one of {4, 8}. Default
        is 8.
    random_state : int, optional
        The seed to use for random number generation. Useful for reproducing
        generated data. If None then a random seed is chosen. Defaults to None.

    Returns
    -------
    Q : scipy.sparse.coo_matrix
        Spatial precision matrix
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
    alpha : np.ndarray
        True values of coefficients of detection covariates.
    beta : np.ndarray
        True values of coefficients of occupancy covariates.
    tau : np.ndarray
        True value of the precision parameter
    z : np.ndarray
        True occupancy state for all `n` sites.

    Raises
    ------
    ValueError
        When `n` is less than the default 150 sites.
        When `min_v` is less than 1.
        When `max_v` is less than 2 or greater than `n`.
        When `ns` is not a positive integer or greater than `n`.
    """
    rng = get_generator(random_state)

    if n < 150:
        raise ValueError('n cant be lower than 150')

    if min_v is None:
        min_v = 2
    elif min_v < 1:
        raise ValueError('min_v needs to be at least 1')

    if max_v is None:
        max_v = n // 10
    elif max_v < 2:
        raise ValueError('max_v is too small')
    elif max_v > n:
        raise ValueError('max_v cant be more than n')

    if ns is None:
        ns = n // 2
    elif ns == 0:
        raise ValueError('ns should be positive')
    elif ns > n:
        raise ValueError('ns cant be more than n')

    surveyed_sites = rng.choice(range(n), size=ns, replace=False)
    visits_per_site = rng.integers(min_v, max_v, size=ns, endpoint=True)

    alpha = rng.standard_normal(q)
    beta = rng.standard_normal(p)
    tau = rng.uniform(*tau_range)

    factors = []
    for i in range(3, n):
        if (n % i) == 0:
            factors.append(i)

    row = rng.choice(factors)
    col = n // row

    Q = rand_precision_mat(row, col, max_neighbors=max_neighbors).astype(float)
    Q_pinv = np.linalg.pinv(Q.toarray())
    eta = rng.multivariate_normal(np.zeros(n), Q_pinv / tau, method='eigh')

    X = rng.uniform(-2, 2, n * p).reshape(n, -1)
    X[:, 0] = 1

    psi = np.exp(-np.logaddexp(0, -X @ beta + eta))
    z = rng.binomial(1, p=psi, size=n)

    W, y = {}, {}
    for i, j in zip(surveyed_sites, visits_per_site):
        _W = rng.uniform(-2, 2, size=j * q).reshape(j, -1)
        _W[:, 0] = 1
        d = np.exp(-np.logaddexp(0, -_W @ alpha))
        W[i] = _W
        y[i] = rng.binomial(1, z[i] * d)

    return Q, W, X, y, alpha, beta, tau, z
