.. occuspytial documentation master file, created by
   sphinx-quickstart on Mon Jul  6 20:06:12 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/zoj613/Occuspytial

.. |codecov| image:: https://codecov.io/gh/zoj613/OccuSpytial/branch/master/graph/badge.svg?style=shield
.. |Documentation| image:: https://readthedocs.org/projects/occuspytial/badge/?version=latest
.. |circleci| image:: https://circleci.com/gh/circleci/circleci-docs.svg?style=shield
.. |pypi| image:: https://img.shields.io/pypi/pyversions/OccuSpytial
.. |pypiver| image:: https://img.shields.io/pypi/v/OccuSpytial
.. |licensing| image:: https://img.shields.io/pypi/l/OccuSpytial

Welcome to OccuSpytial's documentation!
=======================================

|codecov|  |Documentation|  |circleci|  |pypi|  |pypiver|  |licensing|


``OccuSpytial`` is a library for performing bayesian inference of single-season
site occupancy models. A species occupancy model is used to account for imperfect
detection of a species in surveys and to determine the probability of occupancy
:math:`(\psi_i)` at each site. This is done by quantifying the conditional
detection probability :math:`(d_{ij})` of a species at a site based off of data.
This library specifically implements models that take into account the spatial
autocorrelation between between neighboring sites for the occupancy covariates.

The basic formulation of the model as shown by `Rorazio and Royle (2008) <https://tinyurl.com/y7pb4sg6>`_ is:

.. math::

   \begin{align}
      \begin{split}
         z_i|\psi_i \sim& \text{ Bernoulli}(\psi_i)\\
         y_{ij}|z_i, d_{ij} \sim& \text{ Bernoulli}(z_id_{ij}).
      \end{split}
   \end{align}

The probabilities :math:`\psi_i` and :math:`d_{ij}` are linked to vectors
:math:`\mathbf{\alpha}` and :math:`\mathbf{\beta}` via a suitable link function
:math:`f(.)` such that :math:`\psi_i = f(\mathbf{x}_i^T\mathbf{\beta} + \eta_i)`
and :math:`d_{ij} = f(\mathbf{w}_{ij}^T\mathbf{\alpha})`; where :math:`\eta_i`
is the spatial random effect used to model the effect that the site neighbourhood
structure has on the species occurrence probabilities, :math:`\mathbf{x}_i` is
site :math:`i`'s covariance, :math:`\mathbf{w}_{ij}` is the detection covariate
of site :math:`i` at the :math:`j`'th visit, :math:`\mathbf{\alpha}` and :math:`\mathbf{\beta}`
are detection and occupancy regression effects. Currently, the spatial random
effects are modelled using an `Intrinsic Conditional Autoregressive (ICAR) <https://www.statsref.com/HTML/car_models.html>`_ model.

For more information about the currently implemented methods, see :ref:`samplers` and
:ref:`userguide` for examples of how to use each. All implemented samplers are fast
and use C code (via Cython) for computationally intensive parts of each algorithm.

.. todo::

   - Implement a Hamiltonian Monte Carlo sampler variant.


.. toctree::
   :maxdepth: 2
   :caption: Installation

   install

.. _userguide:
.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide.ipynb


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   samplers
   distributions
   utils



License
-------

Copyright (c) 2018-2020, Zolisa Bleki and `contributors <https://github.com/zoj613/OccuSpytial/graphs/contributors>`_.

OccuSpytial is free software made available under the BSD License. For details
see the `LICENSE <https://github.com/zoj613/OccuSpytial/blob/master/LICENSE>`_ file.
