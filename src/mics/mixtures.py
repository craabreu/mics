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
from mics.utils import cross_covariance
from mics.utils import info
from mics.utils import mkcallable
from mics.utils import multimap
from mics.utils import overlapSampling
from mics.utils import pinv


class mixture:
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

        info(verbose, "Setting up MICS case:", title)

        m = self.m = len(samples)
        self.samples = samples
        self.verbose = verbose
        S = range(m)
        info(verbose, "Number of samples:", m)

        if m == 0:
            raise ValueError("sample set is empty")

        names = [list(samples[i].dataset.columns.values) for i in S]
        info(verbose, "Properties:", ", ".join(names[0]))

        if any([names[i] != names[0] for i in range(1, m)]):
            raise ValueError("provided samples have distinct properties")

        n = self.n = np.array([samples[i].dataset.shape[0] for i in S])
        info(verbose, "sample sizes:", str(n))

        neff = np.array([samples[i].neff for i in S])
        info(verbose, "Effective sample sizes:", str(neff))

        pi = self.pi = neff/sum(neff)
        info(verbose, "Mixture composition:", pi)

        potentials = [samples[i].potential for i in S]
        u = [multimap(potentials, samples[i].dataset) for i in S]

        f = self.f = overlapSampling(u)
        info(verbose, "Initial free-energy guess:", f)

        self.P = [np.empty([m, n[i]]) for i in S]
        self.pm = [np.empty(m) for i in S]
        self.u0 = [np.empty([1, n[i]]) for i in S]

        iter = 1
        df, B0 = self._newton_raphson_iteration(u)
        while any(abs(df) > tol):
            iter += 1
            f[1:m] += df
            df, B0 = self._newton_raphson_iteration(u)
        info(verbose, "Free energies after %d iterations:" % iter, f)

        iB0 = self.iB0 = pinv(B0)
        S0 = sum(pi[i]**2*covariance(self.P[i], self.pm[i], samples[i].b) for i in S)
        self.Theta = iB0.dot(S0.dot(iB0))
        info(verbose, "Free-energy covariance matrix:", self.Theta)

    # ======================================================================================
    def free_energies(self):
        """
        Returns a data frame containing the relative free energies of the datasetd samples
        of a `mixture`, as well as their standard errors.

        """
        T = self.Theta
        df = np.sqrt([T[i, i] - 2*T[i, 0] + T[0, 0] for i in range(self.m)])
        return pd.DataFrame(data={'f': self.f, 'δf': df})

    # ======================================================================================
    def reweight(self, properties, potential, parameter, combinations=None):
        """
        Performs reweighting of the properties computed by `functions` from the mixture to
        the samples determined by the provided `potential` with all `parameter` values.

        Args:
            properties (function list):
            potential (function):
            parameter (pandas.DataFrame):

        """
        sample = self.samples
        m = self.m
        pi = self.pi
        P = self.P
        pm = self.pm
        S = range(m)
        b = [sample[i].b for i in S]

        functions = [mkcallable(p) for p in properties]

        def compute(x):
            return np.vstack([np.ones(x.shape[0]), multimap(functions, x)])

        z = [compute(sample[i].dataset) for i in S]

        y0 = []
        Xi = []
        for value in parameter:
            info(self.verbose, "Parameter value:", value)
            u = [multimap([lambda x: potential(x, value)], sample[i].dataset) for i in S]
            du = [self.u0[i] - u[i] for i in S]
            maxdu = np.amax([np.amax(du[i]) for i in S])
            y = [np.exp(du[i] - maxdu)*z[i] for i in S]
            ym = [np.mean(y[i], axis=1) for i in S]
            Syy = [covariance(y[i], ym[i], b[i]) for i in S]
            Spy = [cross_covariance(P[i], pm[i], y[i], ym[i], b[i]) for i in S]
            Sy0y0 = sum(pi[i]**2*Syy[i] for i in S)
            Sp0y0 = sum(pi[i]**2*Spy[i] for i in S)
            Z0 = -sum(P[i].dot(y[i].T)*pi[i]/sample[i].n for i in S)
            A = Z0.T.dot(self.iB0.dot(Sp0y0))
            y0.append(sum(pi[i]*ym[i] for i in S))
            Xi.append(Sy0y0 + A + A.T + Z0.T.dot(self.Theta.dot(Z0)))

        print([f.__name__ for f in functions])
        for k in range(len(parameter)):
            print([y0[k][i]/y0[k][0] for i in range(1, len(properties)+1)])

    # ======================================================================================
    def _newton_raphson_iteration(self, u):
        m = self.m
        P = self.P
        pi = self.pi
        S = range(m)
        g = (self.f + np.log(pi)).reshape([m, 1])
        for i in S:
            x = g - u[i]
            xmax = np.amax(x, axis=0)
            numer = np.exp(x - xmax)
            denom = np.sum(numer, axis=0)
            self.P[i] = numer / denom
            self.u0[i] = -(xmax + np.log(denom))
            self.pm[i] = np.mean(P[i], axis=1)

        p0 = sum(pi[i]*self.pm[i] for i in S)
        B0 = np.diag(p0) - sum(P[i].dot(P[i].T)*pi[i]/self.n[i] for i in S)  # Optimize here
        df = np.linalg.solve(B0[1:m, 1:m], (pi - p0)[1:m])
        return df, B0
