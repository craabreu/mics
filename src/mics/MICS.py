# -*- coding: utf-8 -*-
"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
from numpy.linalg import multi_dot

from mics.mixtures import mixture
from mics.utils import covariance
from mics.utils import cross_covariance
from mics.utils import info
from mics.utils import pinv


class MICS(mixture):
    """A mixture of independently collected samples (MICS)

        Args:
            samples (list or tuple):
                a list of samples.
            title (str, optional):
                a title.
            verbose (bool, optional):
                a verbosity tag.
            tol (float, optional):
                a tolerance.

    """

    # ======================================================================================
    def __init__(self, samples, title="Untitled", verbose=False, tol=1.0E-8):

        m, n, neff = self._definitions(samples, title, verbose)

        b = self.b = [s.b for s in samples]
        pi = self.pi = neff/sum(neff)
        info(verbose, "Mixture composition:", pi)

        P = self.P = [np.empty([m, k], np.float64) for k in self.n]
        pm = self.pm = [np.empty(m, np.float64) for i in range(m)]
        self.u0 = [np.empty([1, n], np.float64) for n in self.n]

        iter = 1
        df = self._newton_raphson_iteration()
        while any(abs(df) > tol):
            iter += 1
            self.f[1:m] += df
            df = self._newton_raphson_iteration()
        info(verbose, "Free energies after %d iterations:" % iter, self.f)

        S0 = sum(pi[i]**2*covariance(P[i], pm[i], b[i]) for i in range(m))
        self.Theta = multi_dot([self.iB0, S0, self.iB0])
        info(verbose, "Free-energy covariance matrix:", self.Theta)

    # ======================================================================================
    def _newton_raphson_iteration(self):
        m = self.m
        u = self.u
        P = self.P
        pi = self.pi
        g = (self.f + np.log(pi)).reshape([m, 1])
        S = range(m)
        for i in S:
            x = g - u[i]
            xmax = np.amax(x, axis=0)
            numer = np.exp(x - xmax)
            denom = np.sum(numer, axis=0)
            self.P[i] = numer / denom
            self.u0[i] = -(xmax + np.log(denom))
            self.pm[i] = np.mean(P[i], axis=1)

        p0 = sum(pi[i]*self.pm[i] for i in S)
        B0 = np.diag(p0) - sum(P[i].dot(P[i].T)*pi[i]/self.n[i] for i in S)  # Use BLAS here
        self.iB0 = pinv(B0)
        df = np.matmul(self.iB0, pi - p0)
        return df[1:m]-df[0]

    # ======================================================================================
    def _reweight(self, u, z):
        S = range(self.m)
        pi = self.pi
        P = self.P
        pm = self.pm

        a = [np.exp(self.u0[i] - u[i]) for i in S]
        y = [np.vstack([a[i]*z[i], a[i]]) for i in S]
        ym = [np.mean(y[i], axis=1) for i in S]

        y0 = sum(pi[i]*ym[i] for i in S)

        Syy = [covariance(y[i], ym[i], self.b[i]) for i in S]
        Sy0y0 = sum(pi[i]**2*Syy[i] for i in S)
        Spy = [cross_covariance(P[i], pm[i], y[i], ym[i], self.b[i]) for i in S]
        Sp0y0 = sum(pi[i]**2*Spy[i] for i in S)
        Z0 = -sum(np.matmul(P[i], y[i].T)*pi[i]/self.n[i] for i in S)
        M = multi_dot([Z0.T, self.iB0, Sp0y0])
        Xi = Sy0y0 + M + M.T + multi_dot([Z0.T, self.Theta, Z0])

        Q = y0[-1]
        A = y0/Q
        A[-1] = -np.log(Q)

        Sigma = Xi
        return A[-1], Xi[-1, -1], A, Sigma
