OccuSpytial
-----------

A package for fast bayesian analysis of spatial occupancy models. OccuSpytial implements an efficient Gibbs Sampler for the single season site spatial occupancy model using the Intrinsic Conditional Autoregressive (ICAR) model for spatial random effects. The Gibbs sampler is made possible by using Polya-Gamma data-augmentation to obtain closed form expressions for the full conditional distributions of the parameters of interest.

Installation
------------
currently this package can be installed by downloading the repository and running the following command on the folder with the package:
::
  python setup.py install --user


Usage
-----
The initializing `Sampler` class accepts:

* a 2-D numpy array :math:`X` (occupancy effects design matrix).
* a python dictionary object :math:`W` (each key-value entry is the site number and detection effects design matrix for that particular site ).
* a python dictionary object :math:`y` (each key-value entry is the site number and an array containing the detection/non-detection info of that particular site).
* a 2-D numpy / sparse-matrix :math: `Q` (the ICAR model precision matrix).
* Only 2 models are supported currently, namely ICAR and RSR (Reduced Spatial Regression).
::
    >>> from occuspytial.interface import Sampler
    >>> import numpy as np
    >>> from datamodule import X, W, y, Q  # datamodule is the file with all the arrays X, W, y and Q
    # the dictionary with hyperparameters for the spatial occupancy model
    >>> HYPERS = {
          "a_mu": np.zeros(W[0].shape[1], dtype='float64'), 
          "a_prec": np.diag([1. / 1000] * W[0].shape[1]),
          "b_mu": np.zeros(X.shape[1], dtype='float64'),
          "b_prec": np.diag([1. / 1000] * X.shape[1]),
          "i_1": 0.5,
          "i_2": 0.0005,
        }
    # the dictionary with initial values for parameters alpha, beta, tau & eta
    >>> INIT = {
          "alpha": np.array([0, 0.]),
          "beta": np.array([0., 0., 0]),
          "tau": 1,
          "eta": np.random.uniform(0, 1, size=Q.shape[0])
        }
    # initialize the sampler setting the number of chains to 3 and using the icar model for the spatial random effects
    # note that the chains are ran simultanously in parallel.
    >>> icarmodel = Sampler(X, W, y, Q, INIT, HYPERS, model='icar', chains=3)
    # keep only the last 300 iterations from each of the 3 chains to form a chain with 900 samples per parameter.
    # progressbar=True is set so that the progress bar can be shown while the sampler is running.
    >>> icarmodel.run(iters=100000, burnin=99700, progressbar=True)
    # print the summary table containing the posterior estimates of the parameters, their standard errors and convergence diagnostics info
    >>> print(icarmodel.summary())
                        mean       std      2.5%     97.5%      R_hat  Geweke_score
        $\alpha_0$ -0.015179  0.057961 -0.125347  0.091161   1.003988     -1.333337
        $\alpha_1$  1.734395  0.071640  1.597810  1.877713   1.003778      2.369414
        $\beta_0$   0.156570  0.397243 -0.608351  0.930774   1.002827     -0.678714
        $\beta_1$   2.703757  0.576002  1.704926  3.920980   1.017302      1.098281
        $\beta_2$  -1.997502  0.518368 -3.153621 -1.100192   1.003005      0.339891
        $\tau$      0.885988  0.344655  0.388398  1.781189   0.999137      0.697562
    >>> icarmodel.trace_plots(show=True) # display the traceplots of the parameters alpha, beta, and tau.
    >>> icarmodel.corr_plots(show=True) # display the correlation plots of the parameters alpha, beta, and tau.
    
