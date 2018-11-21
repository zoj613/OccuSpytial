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
from __future__ import print_function
from datetime import timedelta
import sys, time

import numpy as np
from numpy.random import standard_normal as std_norm
from scipy.linalg import cholesky as chol
from scipy.sparse import issparse
try:
    from sksparse.cholmod import cholesky as sp_chol
except ImportError:
    pass

try:
    FILE = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
except:
    FILE = open('temp.txt', mode='w', encoding='utf8', buffering=1)


__all__ = [
    "track_progress",
    "affine_sample",
    "CustomDict"
]


def _progressbar(current, total, elapsed, remain, _fill="█"): 
    
    progress = current / total
    BAR_LENGTH = 25 # Modify this to change the length of the progress bar
    block = int(round(BAR_LENGTH * progress))
    _input = [
        _fill * block + "-" * (BAR_LENGTH - block), #'█'
        progress * 100,
        str(elapsed).split(".")[0],
        current,
        total,
        str(remain).split(".")[0],
        round(current / elapsed.total_seconds(), 2)
    ]
    text = '\r{1:.1f}%[{0}] {3}/{4} [{2}<{5}, {6}draws/s] '.format(*_input)
    print(text, end="", file=FILE, flush=True)
    if progress == 1:
        print("\r\n")


def track_progress(i, start, iters):
    
    now = time.monotonic()
    elapsed = timedelta(seconds=now - start)
    est_total = timedelta(seconds=iters/(i + 1) * (now - start))
    remain = est_total - elapsed
    _progressbar(i + 1, iters, elapsed, remain)


def affine_sample(mean, cov, return_factor=False):
    """ Function doc """
    try:
        factor = sp_chol(cov)
        x = mean + factor.L().dot(std_norm(len(mean)))
    
    except (NameError, AttributeError):

        # sp_chol failed to import and/or cov is dense
        cov_dense = cov.toarray() if issparse(cov) else cov
        factor = chol(cov_dense, lower=True, check_finite=False)
        x = mean + factor.dot(std_norm(len(mean)))   
    
    finally:
        
        if return_factor:
           return x, factor
        else:
           return x


def acf(x, lag=0):
    """ Function doc """
    if lag == 0:
        return 1
    elif lag < len(x) - 1:    
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    else:
        raise Exception("lag must be less than {}".format(len(x) - 1))


class CustomDict(dict):    
    # a mix-in class for group-indexing W and y dictionaries
    def slice(self, *keys):
        try:
            out = [self[k] for k in keys]
        except TypeError: #if input is not hashable
            out = [self[k] for k in keys[0]]
        return np.concatenate(tuple(out))

        
class SpatialStructure(object):
    """ A class intended for generating spatial precision matrix used in 
    models like CAR, ICAR and RSR"""
    
    def __init__ (self, n):
        """ Class initialiser """
        self.n = n
        self.lattice = None
        self.A = None
        
    def _generate_random_lattice(self, n=None):
        """ Function doc """
        if n is not None:
            a = n
        else:
            a = self.n

        b = np.arange(1, a + 1)
        c = b[a % b == 0][1:-1]  # multiplicative factors of n except for 1
        d = []
        for i in c:
            out = c[i * c == a][0]  # factor whose product with i equals a
            d.append((i, out))
        
        d = d[-(a // 2):]  # remove duplicate pairs
        factors = d[np.random.randint(0,len(d))]  # randomly select one element
        # create a lattice rectangular grid of dimensions factors[0] x factors[1]
        self.lattice = np.arange(1, a + 1).reshape(factors)
    
    def _neighbor_indx(self, indx, n_type=4):
        """ Function doc """
        assert n_type in [4, 8], "n_type must be 4 or 8"
        out = []
        
        if n_type == 8:
            out.append((indx[0] - 1, indx[1] - 1))  # north west
            out.append((indx[0] + 1, indx[1] - 1))  # south west
            out.append((indx[0] - 1, indx[1] + 1))  # north east
            out.append((indx[0] + 1, indx[1] + 1))  # south east
        
        out.append((indx[0], indx[1] - 1))
        out.append((indx[0], indx[1] + 1))
        out.append((indx[0] - 1, indx[1]))
        out.append((indx[0] + 1, indx[1]))

        return out
    
    def _adjacency_matrix(self, n_type=4):
        """ use the generated lattice to create an adjacency matrix A, where
        an element A[i, j] = 1 if i and j are neighbors and A[i, j] = 0 
        otherwise for i =/= j. A[i, i] = 0 for all i."""
        A = np.zeros((self.n, self.n))
        for indx, site in np.ndenumerate(self.lattice):
            A[site - 1, site - 1] = 0 
            # randomly decide the maximum number of neighbors for site.
            if n_type == 'mixed':
                
                type_of_neighbor = np.random.choice([4, 8], p=[0.5, 0.5])
                neighbor_indx = self._neighbor_indx(indx, type_of_neighbor)
            else:
                
                neighbor_indx = self._neighbor_indx(indx, n_type)

            for row, col in neighbor_indx:
                try:
                    neighbor_site = self.lattice[row, col]
                    A[site - 1, neighbor_site - 1] = 1
                    A[neighbor_site - 1, site - 1] = 1
                except IndexError:
                    continue
        self.A = A
            
    def spatial_precision(self, n_type='mixed', rho=1):
        """ Function doc """
        self._generate_random_lattice()
        self._adjacency_matrix(n_type)
        D = np.diag(self.A.sum(axis=0))
        return D - rho * self.A
