.. _samplers:

Gibbs Samplers
==============

.. currentmodule:: occuspytial.gibbs

.. autosummary::
    :nosignatures:

    ~base.GibbsBase
    ~logit.LogitICARGibbs
    ~logit.LogitRSRGibbs
    ~probit.ProbitRSRGibbs


Base Sampler
------------
.. autoclass:: occuspytial.gibbs.base.GibbsBase
    :members:


Logit link based
----------------
.. autoclass:: occuspytial.gibbs.logit.LogitICARGibbs
    :members:
    :show-inheritance:
    :undoc-members: step
.. autoclass:: occuspytial.gibbs.logit.LogitRSRGibbs
    :members:
    :show-inheritance:
    :undoc-members: step


Probit link based
-----------------
.. autoclass:: occuspytial.gibbs.probit.ProbitRSRGibbs
    :members:
    :show-inheritance:
    :undoc-members: step
