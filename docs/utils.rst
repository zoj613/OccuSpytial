Utility Functions & Classes
===========================

Several development utilities and other functions not part of the sampling API
are included here.


Datasets
--------

Utilities for generating data used for testing and exploration of provided samplers.


.. autofunction:: occuspytial.utils.rand_precision_mat

.. autofunction:: occuspytial.utils.make_data


Development Utils
-----------------

.. autofunction:: occuspytial.utils.get_generator

.. autofunction:: occuspytial.gibbs.parallel.sample_parallel

.. autoclass:: occuspytial.data.Data
    :members:
    :special-members: __getitem__

.. autoclass:: occuspytial.posterior.PosteriorParameter
    :members:

.. autoclass:: occuspytial.chain.Chain
    :members:

.. autoclass:: occuspytial.gibbs.state.State
    :members:

.. autoclass:: occuspytial.gibbs.state.FixedState
    :members:

