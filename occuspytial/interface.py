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
from multiprocessing import cpu_count
import sys
from warnings import simplefilter

from beautifultable import BeautifulTable
from loky import get_reusable_executor
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch as specdensity
from scipy.stats import gaussian_kde

from .icar.model import ICAR
from .utils.utils import acf

simplefilter('ignore', UserWarning)
plt.style.use('ggplot')


class Sampler(object):
    """ Class doc """
    
    def __init__(self, X, W, y, Q, INIT, HYPERS, model='icar', 
                  chains=2, threshold=0):

        self.mode = model
        self.n_chains = chains
        self.inits = [INIT]
        self._new_inits(INIT) # set initial values for the additional chains
        if model.lower() == 'icar':
            self.model = ICAR(X, W, y, Q, INIT, HYPERS)
        elif model.lower() == 'rsr':
            self.model = ICAR(X, W, y, Q, INIT, HYPERS, use_rsr=True, threshold=threshold)
        else:
            raise Exception("model choice can only be 'icar' or 'rsr'")
        self._names = self.model._names
        self.fullchain = np.array(self._names, ndmin=2)


    def _new_inits(self, init):
        """ Function doc """
        self.inits = [init]
        for _ in range(1, self.n_chains):
            # create multiple initial values for the additional chains
            # using random pertubation of the user supplied initial values
            _init = {}
            for key, value in init.items():
                if key == "alpha" or key == "beta":
                    newvalue = value + np.random.uniform(-2, 2, len(value))
                elif key == "tau":
                    newvalue = value + np.random.uniform(0, 5)
                else:
                    newvalue = value + np.random.uniform(-2, 2, len(value))
    
                _init[key] = newvalue
            self.inits.append(_init)
 
    def _get_samples(self, args):
        """ Function doc """
        model, iters, burnin, init, progressbar = args
        model.run_sampler(iters, burnin, init, progressbar=progressbar)
        return model._traces
        
            
    def run(self, iters=1000, burnin=None, new_init=None, progressbar=True):

        executor = get_reusable_executor(max_workers=cpu_count())

        if new_init is not None:
            self._new_inits(new_init)

        args = [(self.model, iters, burnin, init, progressbar) for init in self.inits]
        results = executor.map(self._get_samples, args)
        for i in range(self.n_chains):
            out = next(results)
            setattr(Sampler, 'chain{}'.format(i), out)
            self.fullchain = np.concatenate((self.fullchain, out))


    def trace_plots(self, show=True, save=False):

        traces = self.fullchain
        plot_rows = traces.shape[1]
        plt.figure(figsize=(6, 8))
        for i in range(plot_rows):
            data = traces[:, i][1:].astype(np.float64)
            plt.subplot(plot_rows, 2, 2*i + 1)
            plt.plot(data)
            plt.title(self._names[i])
            plt.subplot(plot_rows, 2, 2*i + 2)
            
            s_data = sorted(data)
            plt.plot(s_data, gaussian_kde(data).pdf(s_data), linewidth=2)
            plt.hist(data, bins=55, density=True, histtype='stepfilled', color='red', alpha=0.3)
            plt.ylabel('')
        plt.tight_layout()
        if save:
            plt.savefig('traces.svg', format='svg', bbox_inches='tight')
        plt.show() if show else plt.clf()


    def corr_plots(self, num_lags=50, show=True, save=False):

        traces = self.fullchain
        plot_rows = traces.shape[1]
        plt.figure(figsize=(6, 8))
        for i in range(plot_rows):
            data = traces[:, i][1:].astype(np.float64)
            lagdata = [acf(data, lag=i) for i in range(0, num_lags + 1)]
            sub = plt.subplot(
                plot_rows, 
                1, i + 1, 
                xlim=(-1, num_lags + 1), 
                ylim=(min(lagdata) - 0.2 if min(lagdata) < 0 else -0.2 , 1.2)
            )
            plt.plot(lagdata, 'C0o', markersize=5)
            ymaxs = [y - 0.05 if y > 0 else y + 0.05 for y in lagdata]
            plt.vlines(np.arange(num_lags + 1), ymin=0, ymax=ymaxs, color='k')
            plt.hlines(y=0, xmin=-1, xmax=num_lags + 1, color='C0')
            plt.title("acf of {}".format(self._names[i]))
        plt.tight_layout()
        if save:
            plt.savefig('corr.svg', format='svg', bbox_inches='tight')
        plt.show() if show else plt.clf()


    def gelman(self, chain):
        """ Function doc """
        if self.n_chains == 1:
            raise Exception("the number of chains needs to be 2 or more.")
        else:
            # split up the big chain into multiple chains
            try:
                s = np.split(chain, self.n_chains, axis=0)
            except ValueError:
                raise Exception("chains need to be of equal length. \
                                Lower or increase the number of chains.") 
            M = self.n_chains  # number of chains
            N = s[0].shape[0] # length of each chain (assume all are of equal length)
            # squared differences of each chain
            sqd = [(s[i].mean(axis=0) - chain.mean(axis=0))**2 for i in range(M)]
            # calculate the between chains variances for each parameter
            B = np.stack(sqd).sum(axis=0) * s[0].shape[0] / (len(s) - 1)
            
            # sample variance of each chain
            sv = [np.var(s[i], axis=0, ddof=1) for i in range(M)]
            # within-chain variances  for each parameter
            W = np.stack(sv).mean(axis=0)
            
            # the pooled variance
            V = (N - 1) * W / N + (M + 1) * B / (M * N)
            
            return np.sqrt(V / W)        


    def geweke(self, chain, first=0.1, last=0.5):
        """ Function doc """
        if self.n_chains == 1:
            raise Exception("the number of chains needs to be 2 or more.")
        else:
            x1 = chain[:int(first * 100)]
            x2 = chain[int((1 - last) * 100):]
            n1 = x1.shape[0]
            n2 = x2.shape[0]
            x1mean = x1.mean(axis=0)
            x2mean = x2.mean(axis=0)
            
            num_of_params = chain.shape[1]
            s1 = np.empty(num_of_params)
            s2 = s1
            
            for i in range(num_of_params):
                s1[i] = specdensity(x1[:, i])[1][0]
                s2[i] = specdensity(x2[:, i])[1][0]
                
            return (x1mean - x2mean) / np.sqrt(s1 / n1 + s2 / n2)

    
    @property
    def summary(self):
        
        table = BeautifulTable()
        table.column_headers = [
            'param', 'est. mean', 'std error', 'cred. 2.5%', 'cred. 97.5%', 'PSRF', 'geweke stat'
        ]
        table.set_style(BeautifulTable.STYLE_NONE)
        
        fullchain = self.fullchain[1:].astype(np.float64)
        rhat = self.gelman(fullchain)
        gewe = self.geweke(fullchain)
        means = fullchain.mean(axis=0)
        stds = fullchain.std(axis=0, ddof=1)
        lower = np.quantile(fullchain, axis=0, q=0.025)
        upper = np.quantile(fullchain, axis=0, q=0.975)
        
        for i, param in enumerate(self._names):
            table.append_row(
                [param] + [means[i], stds[i], lower[i], upper[i], rhat[i], gewe[i]]
            )
            
        return table
