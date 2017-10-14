"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np

from mics.utils import info
from mics.utils import multimap
from mics.utils import overlapSampling


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

        n = self.n = [states[i].sample.shape[0] for i in S]
        info(verbose, "Sample sizes:", n)

        neff = np.array([states[i].neff for i in S])
        info(verbose, "Effective sample sizes:", neff)

        pi = self.pi = neff/sum(neff)
        info(verbose, "Mixture composition:", pi)

        potentials = [states[i].potential for i in S]
        u = [multimap(potentials, states[i].sample) for i in S]

        f = self.f = overlapSampling(u)
        info(verbose, "Initial free-energy guess:", f)

        P = self.P = [np.empty([m, n[i]]) for i in S]
        self.u0 = [np.empty([1, n[i]]) for i in S]

        def iteration():
            self._compute(f, u)
            pm = [np.mean(P[i], axis=1) for i in S]
            p0 = sum(pi[i]*pm[i] for i in S)
            B0 = np.diag(p0) - sum(P[i].dot(P[i].T)*pi[i]/n[i] for i in S)  # Optimize here
            df = np.linalg.solve(B0[1:m, 1:m], (pi - p0)[1:m])
            return df, p0, B0

        iter = 0
        df, p0, B0 = iteration()
        while any(abs(df) > tol):
            iter += 1
            f[1:m] += df
            df, p0, B0 = iteration()
        info(verbose, "Free energies after %d iterations:" % iter, f)

    def _compute(self, f, u):
        """
        Computes probabilities
        """
        m = len(u)
        g = (f + np.log(self.pi)).reshape([m, 1])
        for i in range(m):
            x = g - u[i]
            xmax = np.amax(x, axis=0)
            numer = np.exp(x - xmax)
            denom = np.sum(numer, axis=0)
            self.P[i] = numer / denom
            self.u0[i] = -(xmax + np.log(denom))

#     def means(self, X):
#         """
#         Computes the means of a set of properties
#         """
#         m = len(X)
#         n = X[0].shape[0]
#         xm = [np.empty(m)
#         for i in range(len(X)):
#             x
