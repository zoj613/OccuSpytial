OccuSpytial
-----------

A package for fast bayesian analysis of spatial occupancy models. OccuSpytial implements an efficient Gibbs Sampler for the single season site spatial occupancy model using the Intrinsic Conditional Autoregressive (ICAR) model for spatial random effects. The Gibbs sampler is made possible by using Polya-Gamma data-augmentation to obtain closed form expressions for the full conditional distributions of the parameters of interest when using the logit link function to model the occupancy and detection probabilities. Multiple chains of the Gibbs sampler are ran in parallel.

Installation
------------
currently this package can be installed by downloading the repository and running the following command on the folder with the package:
```
   git clone https://github.com/zoj613/OccuSpytial.git
   cd OccuSpytial
   python setup.py install
```
 
It is strongly recommended that you have the package `scikit-sparse` installed before using this package in order to fully take advantage of the speed gains possible. Using this package with `scikit-sparse` installed results in significant speedup during sampling.

Testing
-------
To run tests after installation, the package `pytest` is required. Simply run
the following line from the terminal in this package's root directory.
```
python -m pytest
```
If all tests pass, then you're good to go.

Usage
-----
The initializing `Sampler` class accepts:

* a 2-D numpy array `X` (occupancy effects design matrix).
* a python dictionary object `W` (each key-value entry is the site number and detection effects design matrix for that particular site ).
* a python dictionary object `y` (each key-value entry is the site number and an array containing the detection/non-detection info of that particular site).
* a 2-D numpy / sparse-matrix `Q` (the ICAR model precision matrix).
* Only 2 models are supported currently, namely ICAR and RSR (Reduced Spatial Regression).
```python
    from occuspytial.interface import Sampler
    import numpy as np
    from datamodule import X, W, y, Q  # datamodule is the file with all the arrays X, W, y and Q
    # the dictionary with hyperparameters for the spatial occupancy model
    HYPERS = dict(
        a_mu=np.zeros(W[0].shape[1]),
        a_prec=np.diag([1. / 1000] * W[0].shape[1]),
        b_mu=np.zeros(X.shape[1]),
        b_prec=np.diag([1. / 1000] * X.shape[1]),
        i_1=0.5,
        i_2=0.0005
    )
    # the dictionary with initial values for parameters alpha, beta, tau & eta
    INIT = {
        "alpha": np.array([0, 0.]),
        "beta": np.array([0., 0., 0]),
        "tau": 1,
        "eta": np.random.uniform(-10, 10, size=Q.shape[0])
    }
    # initialize the sampler setting the number of chains to 3 and using the icar model for the spatial random effects
    # note that the chains are ran simultanously in parallel.
    icarmodel = Sampler(X, W, y, Q, INIT, HYPERS, model='icar', chains=3)
    # keep only the last 300 iterations from each of the 3 chains to form a chain with 900 samples per parameter.
    # progressbar=True is set so that the progress bar can be shown while the sampler is running.
    icarmodel.run(iters=100000, burnin=99700, progressbar=True)
    # print the summary table containing the posterior estimates of the parameters, their standard errors and convergence diagnostics info
    print(icarmodel.summary)
    print(icarmodel.summary[0][1])  # indexing is supported.
    icarmodel.trace_plots(show=True) # display the traceplots of the parameters 
    icarmodel.corr_plots(show=True) # display the correlation plots
 ```
TO DO
-----
* Add a folder with dummy data and a usage example notebook.
