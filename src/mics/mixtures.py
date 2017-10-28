# -*- coding: utf-8 -*-
"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
import pandas as pd
from numpy.linalg import multi_dot

from mics.evaluation import genfunc
from mics.evaluation import multimap
from mics.utils import covariance
from mics.utils import cross_covariance
from mics.utils import info
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

        self.samples = samples
        self.title = title
        self.verbose = verbose

        m = self.m = len(samples)
        if m == 0:
            raise ValueError("list of samples is empty")
        info(verbose, "Number of samples:", m)

        names = self.names = list(samples[0].dataset.columns)
        S = range(m)
        if any(list(samples[i].dataset.columns) != names for i in S):
            raise ValueError("provided samples have distinct properties")
        info(verbose, "Properties:", ", ".join(names))

        n = self.n = np.array([samples[i].dataset.shape[0] for i in S])
        self.b = [samples[i].b for i in S]
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
    def reweighting(self,
                    potential,
                    properties={},
                    combinations={},
                    conditions=pd.DataFrame(),
                    **kwargs):
        """
        Performs reweighting of the properties computed by `functions` from the mixture to
        the samples determined by the provided `potential` with all `parameter` values.

        Args:
            potential (function/string):
            properties (dict of functions/strings):
            combinations (dict of strings):
            parameter (pandas.DataFrame):
            **kwargs:

        """
        n = self.n
        pi = self.pi
        P = self.P
        pm = self.pm
        datasets = [s.dataset for s in self.samples]
        S = range(self.m)

        if properties:
            functions = [genfunc(p, self.names, **kwargs) for p in properties.values()]
            z = [np.vstack([np.ones(len(x)), multimap(functions, x)]) for x in datasets]
        else:
            z = [np.ones(n) for n in self.n]

        N = len(conditions)
        f = np.empty(N, np.float64)
        df = np.empty(N, np.float64)
        for j, row in conditions.iterrows():
            condition = row.to_dict()
            info(self.verbose, "Condition[%d]:" % j, condition)
            condition.update(kwargs)

            potfunc = genfunc(potential, self.names, **condition)
            u = [multimap([potfunc], x) for x in datasets]
            y = [np.exp(self.u0[i] - u[i])*z[i] for i in S]
            ym = [np.mean(y[i], axis=1) for i in S]

            y0 = sum(pi[i]*ym[i] for i in S)
            f[j] = -np.log(y0[0])

            Syy = [covariance(y[i], ym[i], self.b[i]) for i in S]
            Sy0y0 = sum(pi[i]**2*Syy[i] for i in S)
            Spy = [cross_covariance(P[i], pm[i], y[i], ym[i], self.b[i]) for i in S]
            Sp0y0 = sum(pi[i]**2*Spy[i] for i in S)
            Z0 = -sum(np.matmul(P[i], y[i].T)*pi[i]/self.n[i] for i in S)
            A = multi_dot([Z0.T, self.iB0, Sp0y0])
            Xi = Sy0y0 + A + A.T + multi_dot([Z0.T, self.Theta, Z0])

            df[j] = np.sqrt(Xi[0, 0]/y0[0]**2)

        result = conditions.copy()
        result['f'] = f
        result['δf'] = df
        return result

    # ======================================================================================
    def _newton_raphson_iteration(self, u):
        m = self.m
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
        B0 = np.diag(p0) - sum(P[i].dot(P[i].T)*pi[i]/self.n[i] for i in S)  # Optimize here
        df = np.linalg.solve(B0[1:m, 1:m], (pi - p0)[1:m])
        return df, B0
