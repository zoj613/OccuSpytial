import warnings

import numpy as np
from scipy.sparse import diags, csc_matrix


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
    D = diags(np.array(W.sum(axis=0))[0])
    Q = D - rho * W
    return csc_matrix(Q)
