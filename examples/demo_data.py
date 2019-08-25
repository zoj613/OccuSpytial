"""This script shows a demo of how to generate dummy data to be use with the
sampler.
"""

import numpy as np
from scipy.special import expit

from occuspytial.utils.dataprocessing import SpatialStructure

__all__ = ["X", "W", "y", "Q"]

# lets set the true parameter values for the occupancy and detection processes
true_alpha = np.array([0, 1.75])
true_beta = np.array([-1.25, -1, -0.5])
len_alpha = len(true_alpha)
len_beta = len(true_beta)

n = 1600  # n is the number of sites visited
K_ = int(0.05 * n)  # K_ = number of visits per site
surv_sites = int(0.5 * n) # number of surveyed sites
# To simplify things we assume the number of visits is the same for all sites
K = np.repeat(K_, surv_sites)

# generate a Precision matrix from a random square lattice grid
Q = SpatialStructure(n).spatial_precision(square_lattice=True)

# set true parameter values for tau and eta
true_tau = 0.5
Q_pinv = np.linalg.pinv(Q)
true_eta = np.random.multivariate_normal(np.zeros(n), Q_pinv / true_tau)

# Here we generate the required X design matrix for the occuapancy process
x1 = np.random.uniform(-2.0, 2.0, size=n * (len_beta - 1))
x = (x1 - np.mean(x1)) / np.std(x1, ddof=1)
# X is 2D array with dimensions (n, r)
X = np.ones((n, len_beta), dtype=float)
X[:, 1:len_beta] = x.reshape(n, len_beta-1)

# Generate true_z, the true occupancy state of each site
psi = expit(X @ true_beta + true_eta)
true_z = np.random.binomial(1, p=psi, size=n)

# Generate W_i the design matrices for each site pertaining to the detection
# process. Also generate y, containing the information about detection of the
# species during each visit of a particular surveyed site.
W, d, y = {}, {}, {}
for i in range(K.size):
    w1 = np.random.uniform(-5.0, 5.0, size=K[i] * (len_alpha - 1))
    w = (w1 - np.mean(w1)) / np.std(w1, ddof=1)
    _W = np.ones((K[i], len_alpha), dtype=float)
    # replace all the w_ik covariates with elements of w, except the 1st col
    _W[:, 1:len_alpha] = w.reshape(K[i], len_alpha - 1)
    d[i] = expit(_W @ true_alpha)
    W[i] = _W
    y[i] = np.random.binomial(1, true_z[i] * d[i])
