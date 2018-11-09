#!/usr/bin/env python
# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2018, Zolisa Bleki
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
import sys
import abc

import numpy as np

from .utils import CustomDict

# done so the base class is compatible with both python 2 and 3.
# https://stackoverflow.com/questions/35673474/using-abc-abcmeta-in-a-way-it-is-compatible-both-with-python-2-7-and-python-3-5
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(b'ABC', (), {})

class MCMCModelBase(ABC):

    def __init__(self, X, W, y, init, hypers):

        self.W = W
        self.X = X
        self.y = y
        self._n = X.shape[0]  # number of sites in total
        self._s = len(self.y) # number of surveyed sites
        self._us = self._n - self._s  # number of unsurveyed sites
        self.Xs = X[:self._s] # X sub-matrix for surveyed sites.
        self._V = np.array(
            [self.W[i].shape[0] for i in range(self._s)],
            dtype=np.int64
        )
        self.init = init
        self.hypers = hypers
        self.not_obs = [] #surveyed sites where species is not observed
        # initialize z, the site occupancy state
        self._z = np.ones(self._n, dtype=np.int64)
        for i in range(self._s):
            if not any(self.y[i]):
                self.not_obs.append(i)
                self._z[i] = 0.0
        self.not_obs = np.array(self.not_obs, dtype=np.int64)
        self.no_size = self.not_obs.shape[0]
        #array to store probability updates for sites where species is not obversed
        self._probs = np.ones(self.not_obs.size, dtype=np.float64)
        # stacked W matrix for all sites where species is not observed
        self._W_ = CustomDict(self.W).slice(self.not_obs)

        self._names = []
        for i in range(W[0].shape[1]):
            self._names.append(r"$\alpha_{0}$".format(i))
        for i in range(X.shape[1]):
            self._names.append(r"$\beta_{0}$".format(i))
        # specify the names of the posterior parameters
        self._names.append(r"$\tau$")
        self._alpha = init["alpha"]
        self._beta = init["beta"]
        self._tau = init["tau"]

    @abc.abstractmethod
    def _alpha_update(self):
        pass

    @abc.abstractmethod
    def _beta_update(self):
        pass

    @abc.abstractmethod
    def _z_update(self):
        pass

    @abc.abstractmethod
    def run_sampler(self, iters=2000):
        pass
