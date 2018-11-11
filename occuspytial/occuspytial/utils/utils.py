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
from datetime import timedelta
import sys, time

import numpy as np
from numpy.random import standard_normal as std_norm
from scipy.linalg import cholesky as chol
try:
    from sksparse.cholmod import cholesky as sp_chol
except ImportError:
    pass

try:
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
except:
    sys.stdout = open('temp.txt', mode='w', encoding='utf8', buffering=1)


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
    #print(text, end="", file=FILE, flush=True)
    sys.stdout.write(text)
    sys.stdout.flush()
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


class CustomDict(dict):    
    # a mix-in class for group-indexing W and y dictionaries
    def slice(self, *keys):
        try:
            out = [self[k] for k in keys]
        except TypeError: #if input is not hashable
            out = [self[k] for k in keys[0]]
        return np.concatenate(tuple(out))
        

