import warnings

import numpy as np
from scipy.linalg import pinvh


def get_generator(random_state=None):
    """Get an instance of a numpy random number generator object.

    This instance uses `SFC64 <https://tinyurl.com/y2jtyly7>`_ bitgenerator,
    which is the fastest numpy currently has to offer as of version 1.19.
    This function conveniently instantiates a generator of this kind and should
    be used in all modules.

    Parameters
    ----------
    random_state : {None, int, array_like[ints], numpy.random.SeedSequence}
        A seed to initialize the bitgenerator. Defaults to ``None``.

    Returns
    -------
    numpy.random.Generator
        Instance of numpy's Generator class, which exposes a number of random
        number generating methods.

    Examples
    --------
    >>> from occuspytial.utils import get_generator
    >>> rng = get_generator()
    # The instance can be used to access functions of ``numpy.random``
    >>> rng.standard_normal()
    -0.203  # random
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

    Examples
    --------
    >>> from occuspytial.utils import rand_precision_mat
    >>> Q = rand_precision_mat(10, 5)
    >>> Q
    <50x50 sparse matrix of type '<class 'numpy.int64'>'
            with 364 stored elements in COOrdinate format>
    # The matrix can be converted to numpy format using method ``toarray()``
    >>> Q.toarray()
    array([[ 3, -1,  0, ...,  0,  0,  0],
           [-1,  5, -1, ...,  0,  0,  0],
           [ 0, -1,  5, ...,  0,  0,  0],
           ...,
           [ 0,  0,  0, ...,  5, -1,  0],
           [ 0,  0,  0, ..., -1,  5, -1],
           [ 0,  0,  0, ...,  0, -1,  3]])
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

    Examples
    --------
    >>> from occuspytial.utils import make_data
    >>> Q, W, X, y, alpha, beta, tau, z = make_data()
    >>> Q
    <150x150 sparse matrix of type '<class 'numpy.float64'>'
            with 1144 stored elements in COOrdinate format>
    >>> Q.toarray()
    array([[ 3., -1.,  0., ...,  0.,  0.,  0.],  # random
           [-1.,  5., -1., ...,  0.,  0.,  0.],
           [ 0., -1.,  5., ...,  0.,  0.,  0.],
           ...,
           [ 0.,  0.,  0., ...,  5., -1.,  0.],
           [ 0.,  0.,  0., ..., -1.,  5., -1.],
           [ 0.,  0.,  0., ...,  0., -1.,  3.]])
    >>> W
    {81: array([[ 1.        ,  1.01334565,  0.93150242],  # random
            [ 1.        ,  0.19276808, -1.71939657],
            [ 1.        ,  0.23866531,  0.0559545 ],
            [ 1.        ,  1.36102304,  1.73611887],
            [ 1.        ,  0.47247886,  0.73410589],
            [ 1.        , -1.9018879 ,  0.0097963 ]]),
     131: array([[ 1.        ,  1.67846707, -1.12476746],
            [ 1.        , -1.63131532, -1.32216705],
            [ 1.        , -1.37431173, -0.79734213],
            ...,
     21: array([[ 1.        ,  1.6416734 , -1.91642502],
            [ 1.        ,  0.2256312 , -1.68929118],
            [ 1.        ,  1.36953093,  1.08758129],
            [ 1.        , -1.08029212,  0.40219588]])}
    >>> X
    array([[ 1.        ,  0.71582433,  1.76344395],
           [ 1.        ,  0.8561976 ,  1.0520401 ],
           [ 1.        , -0.28051247,  0.16809809],
           ...,
           [ 1.        ,  0.86702262, -1.18225448],
           [ 1.        , -0.41346399, -0.9633078 ],
           [ 1.        , -0.23182363,  1.69930761]])
    >>> y
    {15: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # random
     81: array([0, 0, 0, 1, 1, 0]),
     ...,
     21: array([0, 1, 0, 0])}
    >>> alpha
    array([-1.43291816, -0.87932413, -1.84927642])  # random
    >>> beta
    array([-0.62084322, -1.09645564, -0.93371374])  # random
    >>> tau
    1.415532667780688  # random
    >>> z
    array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
           1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
           0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
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
    Q_pinv = pinvh(Q.toarray(), cond=1e-5)
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
