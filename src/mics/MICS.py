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
        pm = self.pm = [np.empty(m, np.float64) for k in self.n]
        self.u0 = [np.empty([1, k], np.float64) for k in self.n]

        iter = 1
        df = self._newton_raphson_iteration()
        while any(abs(df) > tol):
            iter += 1
            self.f[1:m] += df
            df = self._newton_raphson_iteration()
        info(verbose, "Free energies after %d iterations:" % iter, self.f)

        Sp0 = self.Sp0 = sum(pi[i]**2*covariance(P[i], pm[i], b[i]) for i in range(m))
        self.Theta = multi_dot([self.iB0, Sp0, self.iB0])
        info(verbose, "Free-energy covariance matrix:", self.Theta)

        self.Overlap = np.stack(pm)
        info(verbose, "Overlap matrix:", self.Overlap)

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

        p0 = self.p0 = sum(pi[i]*self.pm[i] for i in S)
        B0 = np.diag(p0) - sum(pi[i]*np.matmul(P[i], P[i].T)/self.n[i] for i in S)
        self.iB0 = pinv(B0)
        df = np.matmul(self.iB0, pi - p0)
        return df[1:m]-df[0]

    # ======================================================================================
    def _reweight(self, u, y):
        S = range(self.m)
        pi = self.pi

        w = [np.exp(self.u0[i] - u[i]) for i in S]
        z = [w[i]*y[i] for i in S]

        w0 = sum(pi[i]*np.mean(w[i], axis=1) for i in S).item(0)
        yu = sum(pi[i]*np.mean(z[i], axis=1) for i in S)/w0

        # MICS covariance matrix of s0 (using stored covariance matrix of p0):
        r = [np.concatenate((z[i], w[i])) for i in S]
        rm = [np.mean(r[i], axis=1) for i in S]
        s = [np.concatenate((self.P[i], r[i])) for i in S]
        sm = [np.concatenate((self.pm[i], rm[i])) for i in S]
        C0 = sum(pi[i]**2*cross_covariance(s[i], sm[i], r[i], rm[i], self.b[i]) for i in S)
        Sp0r0, Sr0 = np.split(C0, (self.m, ))
        Ss0 = np.block([[self.Sp0, Sp0r0], [Sp0r0.T, Sr0]])

        # Gradient of yu with respect to s0:
        pu = sum(pi[i]*np.mean(w[i]*self.P[i], axis=1) for i in S)/w0
        pytu = sum(pi[i]*np.matmul(self.P[i], z[i].T)/self.n[i] for i in S)/w0
        G = np.concatenate((np.matmul(self.iB0, np.outer(pu, yu) - pytu),
                            np.diag(np.ones(len(yu))/w0),
                            -yu.reshape([1, len(yu)])/w0))

        # Covariance matrix of yu:
        Theta = multi_dot([G.T, Ss0, G])

        return yu, Theta
