# -*- coding: utf-8 -*-
"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
import pandas as pd

from mics.utils import covariance
from mics.utils import info
from mics.utils import multimap
from mics.utils import overlapSampling
from mics.utils import pinv


class Mixture:
    """A mixture of independently collected samples (MICS)

        Args:
            states (list or tuple):
                a list of states.
            title (str, optional):
                a title.
            verbose (bool, optional):
                a verbosity tag.
            tol (float, optional):
                a tolerance.

    """

    def __init__(self, states, title="Untitled", verbose=False, tol=1.0E-8):

        info(verbose, "Setting up MICS case:", title)

        m = self.m = len(states)
        S = range(m)
        info(verbose, "Number of states:", m)

        if m == 0:
            raise ValueError("state set is empty")

        def names(i):
            return list(states[i].sample.columns.values)
        properties = names(0)
        info(verbose, "Properties:", ", ".join(properties))

        if any([names(i) != properties for i in S]):
            raise ValueError("inconsistent data")

        n = self.n = np.array([states[i].sample.shape[0] for i in S])
        info(verbose, "Sample sizes:", str(n))

        neff = np.array([states[i].neff for i in S])
        info(verbose, "Effective sample sizes:", str(neff))

        pi = self.pi = neff.astype(float)/sum(neff)
        info(verbose, "Mixture composition:", pi)

        potentials = [states[i].potential for i in S]
        u = [multimap(potentials, states[i].sample) for i in S]

        f = self.f = overlapSampling(u)
        info(verbose, "Initial free-energy guess:", f)

        self.P = [np.empty([m, n[i]]) for i in S]
        self.u0 = [np.empty([1, n[i]]) for i in S]

        iter = 1
        df, pm, p0, B0 = self._newton_raphson_iteration(u)
        while any(abs(df) > tol):
            iter += 1
            f[1:m] += df
            df, pm, p0, B0 = self._newton_raphson_iteration(u)
        info(verbose, "Free energies after %d iterations:" % iter, f)

        iB0 = self.iB0 = pinv(B0)
        pm = [np.mean(self.P[i], axis=1) for i in S]
        S0 = sum(pi[i]**2*covariance(self.P[i], pm[i], states[i].b) for i in S)
        self.Theta = iB0.dot(S0.dot(iB0))
        info(verbose, "Free-energy covariance matrix:", self.Theta)

    def free_energies(self):
        """
        Returns a data frame containing the relative free energies of the sampled states
        of a `mixture`, as well as their standard errors.

        """
        T = self.Theta
        df = np.sqrt([T[i, i] - 2*T[i, 0] + T[0, 0] for i in range(self.m)])
        return pd.DataFrame(data={'f': self.f, 'δf': df})

    def _newton_raphson_iteration(self, u):
        m = self.m
        P = self.P
        pi = self.pi
        S = range(m)
        g = (self.f + np.log(self.pi)).reshape([m, 1])
        for i in range(m):
            x = g - u[i]
            xmax = np.amax(x, axis=0)
            numer = np.exp(x - xmax)
            denom = np.sum(numer, axis=0)
            self.P[i] = numer / denom
            self.u0[i] = -(xmax + np.log(denom))

        pm = [np.mean(P[i], axis=1) for i in S]
        p0 = sum(pi[i]*pm[i] for i in S)
        B0 = np.diag(p0) - sum(P[i].dot(P[i].T)*pi[i]/self.n[i] for i in S)  # Optimize here
        df = np.linalg.solve(B0[1:m, 1:m], (pi - p0)[1:m])
        return df, pm, p0, B0
