import warnings

import numpy as np


def rand_precision_mat(lat_row, lat_col, max_neighbors=8, rho=1):
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
    n=100,
    min_v=None,
    max_v=None,
    ns=None,
    p=3,
    q=3,
    tau_range=(0.25, 1.5),
    max_neighbors=8,
    random_state=None,
 ):
    rng = np.random.default_rng(np.random.SFC64(random_state))

    if n < 100:
        raise ValueError('n cant be lower than 50')

    if min_v is None:
        min_v = 2
    elif min_v < 1:
        raise ValueError('min_v needs to be at least 1')

    if max_v is None:
        max_v = n // 10
    elif max_v < 2:
        raise ValueError('v is too small')
    elif max_v > n:
        raise ValueError('v cant be more than n')

    if ns is None:
        ns = n // 2
    elif ns == 0:
        raise ValueError('ns should be positive')
    elif ns > n:
        raise ValueError('ns cant be more than n')

    surveyed_sites = rng.choice(range(n), size=ns)
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

    Q = rand_precision_mat(row, col, max_neighbors=max_neighbors)
    Q_pinv = np.linalg.pinv(Q.toarray())
    eta = rng.multivariate_normal(np.zeros(n), Q_pinv / tau, method='cholesky')

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
