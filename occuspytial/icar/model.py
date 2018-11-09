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
import time

import numpy as np
from pandas import DataFrame
from scipy.linalg import eigh, inv, solve_triangular as tri_solve
from scipy.sparse import csc_matrix
from scipy.special import expit  # inverse logit
from pypolyagamma import PyPolyaGamma

from .helpers.ctypesfunc import _num_prod
from ..utils.utils import affine_sample, CustomDict, track_progress
from ..utils.basemodel import MCMCModelBase

pg = PyPolyaGamma()


class ICAR(MCMCModelBase):

    def __init__(self, X, W, y, Q, init, hypers, use_rsr=False, threshold=0.5):

        super().__init__(X, W, y, init, hypers)
        rank = np.linalg.matrix_rank(Q)
        assert rank < self._n, "Q must be singular"
        self._k = self._z - 0.5  # initial value of k = z - 0.5*1
        self._Wc = None
        self._kc = None
        # constant shape parameter for _tau_update method
        self._default_shape = self.hypers["i_1"] + 0.5 * rank
        # attributes specific to omega_b update method
        self._omega_a = None
        self._omega_b = np.empty(self._n)
        self._ones = np.ones_like(self._omega_b)
        self._use_rsr = use_rsr
        if not use_rsr:
            # ICAR model specific attributes
            self.Q = csc_matrix(Q)
            self._eta = self.init["eta"]
        else:
            # RSR model specific attributes
            XTX_inv = inv(self.X.T.dot(self.X), overwrite_a=True, check_finite=False)
            P = -self.X.dot(XTX_inv).dot(self.X.T)
            P[np.diag_indices(self._n)] += 1
            # above 2 lines equivalent to: I - X @ XTX @ XT
            A = -Q
            A[np.diag_indices(self._n)] = 0
            #A = -self.Q + np.diag(np.diag(self.Q))
            Omega = self._n * (P.dot(A).dot(P)) / A.sum()
            # return eigen vectors of Omega and keep first q columns
            # corresponding to the q largest eigenvalues of Omega greater
            # than the specified threshold
            w, v = eigh(Omega, overwrite_a=True, check_finite=False)
            w, v = w[::-1], np.fliplr(v)  # order eigens in descending order
            q = len(w[w >= threshold])  # number of eigenvalues > threshold
            assert q > 0, "Threshold is set too high. Please lower it."
            print("dimension reduced from {0}->{1}".format(Q.shape[0], q))
            self._K = v[:,:q]  # keep first q eigenvectors of ordered eigens
            self.Minv = self._K.T.dot(Q).dot(self._K)
            self._Ks = self._K[:self._s] # K sub-matrix for surveyed sites
            self.Q = self.Minv # replace Q with Minv in the underlying ICAR
            self._shape = q #set new shape parameter using the _shape method
            #theta initial values
            try:
                self._theta = init["theta"]
            except:
                self._theta = np.random.normal(0, 1, q)


    def _omega_a_update(self):

        Wk_alpha = self._Wc.T.dot(self._alpha)
        self._omega_a = np.empty_like(Wk_alpha)
        pg.pgdrawv(np.ones_like(self._omega_a), Wk_alpha, self._omega_a)


    def _omega_b_update(self, vec):
        
        pg.pgdrawv(self._ones, self.X.dot(self._beta) + vec, self._omega_b)

    @property
    def _shape(self):
        
        return self._default_shape
        
    @_shape.setter
    def _shape(self, rank_of_M):
        # useful for resetting the tau shape parameter when using RSR mddel
        self._default_shape = self.hypers["i_1"] + 0.5 * rank_of_M
        
    
    def _tau_update(self, vec):

        rate = 0.5 * vec.dot(self.Q.dot(vec)) + self.hypers["i_2"]
        self._tau = np.random.gamma(shape=self._shape, scale=1 / rate)

    def _eta_update(self):
        #Cong et al method
        prec = self.Q
        prec.data = prec.data * self._tau
        prec.setdiag(prec.diagonal() + self._omega_b)
        b = self._k - self._omega_b * self.X.dot(self._beta)
        s, L = affine_sample(b, prec, return_factor=True)
        v = -s.sum() / self._omega_b.sum()
        t = s + v * self._omega_b
        try:
            x = L.solve_L(t)
            self._eta = L.solve_Lt(x)

        except AttributeError:
            
            self._eta = splu(
                prec, 
                permc_spec='MMD_AT_PLUS_A', 
                options=dict(SymmetricMode=True)
            ).solve(t)

    def _theta_update(self):

        omega = self._omega_b[:self._s]
        prec = self._tau * self.Minv + np.dot(self._Ks.T * omega, self._Ks)
        b = self._Ks.T.dot(self._k[:self._s] - omega * (self.Xs.dot(self._beta)))
        prec_theta, L = affine_sample(b, prec, return_factor=True)
        x = tri_solve(
            L,
            prec_theta,
            lower=True,
            overwrite_b=True,
            check_finite=False
        )
        self._theta = tri_solve(
            L, 
            x, 
            trans=1, 
            lower=True, 
            overwrite_b=True, 
            check_finite=False
        )


    def _Wk_update(self):
        
        z_ind = np.where(self._z[:self._s] == 1)[0]
        self._Wc = CustomDict(self.W).slice(z_ind).T
        self._kc = CustomDict(self.y).slice(z_ind) - 0.5


    def _alpha_update(self):

        a_mu, a_prec = self.hypers["a_mu"], self.hypers["a_prec"]
        prec = np.dot(self._Wc * self._omega_a, self._Wc.T) + a_prec
        b = np.dot(a_prec, a_mu) + np.dot(self._Wc, self._kc)
        prec_alpha, L = affine_sample(b, prec, return_factor=True)  
        x = tri_solve(
            L, 
            prec_alpha, 
            lower=True, 
            overwrite_b=True, 
            check_finite=False
        )
        self._alpha = tri_solve(
            L, 
            x, 
            trans=1, 
            lower=True, 
            overwrite_b=True, 
            check_finite=False
        )


    def _beta_update(self, vec):
        
        omega = self._omega_b[:self._s]
        b_mu, b_prec = self.hypers["b_mu"], self.hypers["b_prec"]
        prec = np.dot(self.Xs.T * omega, self.Xs) + b_prec
        b = np.dot(self.Xs.T, self._k[:self._s] - omega * vec[:self._s]) +\ 
            np.dot(b_prec, b_mu)
        prec_beta, L = affine_sample(b, prec, return_factor=True)  
        x = tri_solve(
            L, 
            prec_beta, 
            lower=True, 
            overwrite_b=True, 
            check_finite=False
        )
        self._beta = tri_solve(
            L, 
            x, 
            trans=1, 
            lower=True, 
            overwrite_b=True, 
            check_finite=False
        )

    #@profile
    def _z_update(self, vec):
        # fastest without suign armadillo prod() function
        occ = expit(self.X[self.not_obs].dot(self._beta) + vec[self.not_obs])
        omd = 1 / (1 + np.exp(self._W_.dot(self._alpha)))
        # calculate the numerator product for each site i \
        # in the expression for the probability of z=1 | y=0
        #_num_product(omd, occ, self.not_obs, self._V, self._probs)
        _num_prod(omd, occ, self.not_obs, self.no_size, self._V, self._probs)
        # _num_product stores the product on the object self._probs.
        # use the stored values to calculate the probability
        self._probs = self._probs / (1 - occ + self._probs)
        # update the occupancy state array by sampling from a Bernoulli with \
        # self._prob as its parameter value.
        self._z[self.not_obs] = np.random.binomial(n=1, p=self._probs)
        
        # sampling for sites not surveyed
        if self._us > 0:
            # calc the probability of z = 1 | site i not surveyed
            us_prob = expit(self.X[-self._us:].dot(self._beta) + vec[-self._us:])
            # update occupancy state by sampling from a Bernoulli(us_prob)
            self._z[-self._us:] = np.random.binomial(n=1, p=us_prob)
        self._k = self._z - 0.5  # update k = _z - 0.5 for next iteration


    def _update_func(self, func, x, y=None):
        
        try:
            args = [x.dot(y), y]
            self._theta_update()
        except TypeError:
            args = [x]
        finally:
            self._omega_b_update(args[0])
            func() # either self._eta or self._theta
            self._beta_update(args[0])
            self._Wk_update()
            self._omega_a_update()
            self._alpha_update()
            self._z_update(args[0])
            self._tau_update(args[-1])


    def _params_update(self, i):
        """ Function doc """
        try:
            self._update_func(self._theta_update, self._K, self._theta)
        except AttributeError:
            self._update_func(self._eta_update, self._eta)
        finally:
            try:
                track_progress(i, self._starttime, self._num_iter)
            except AttributeError: # self._startime is created only if progressbar=True 
                pass # if exception is thrown then it means progressbar is off thus do nothing and continue

    def _new_init(self, init):
        
        self._alpha = init["alpha"]
        self._beta = init["beta"]
        self._tau = init["tau"]
        self._eta = init["eta"]
        if self._use_rsr:
            try:
                self._theta = init["theta"]
            except KeyError:
                self._theta = np.random.normal(0, 1, self.Minv.shape[0])

    def run_sampler(self, iters=2000, burnin=None, init=None, progressbar=True):
        
        if init is not None:
            self._new_init(init)
            
        if burnin is None:
            burnin = int(0.5 * iters) # default burnin value

        if progressbar: 
            setattr(self, '_starttime', time.monotonic())
            setattr(self, '_num_iter', iters)
            
        j = 0
        out = np.zeros((iters - burnin, len(self._names)))
        for i in range(iters):
            self._params_update(i)
            
            if i >= burnin:
                out[j] = np.append(np.append(self._alpha, self._beta), self._tau)
                j += 1

        self._traces = DataFrame(out)
        self._traces.columns = self._names
