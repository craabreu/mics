"""
.. module:: MBAR
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`MBAR`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
from numpy.linalg import multi_dot
from pymbar import mbar

from mics.mixtures import mixture
from mics.utils import info
from mics.utils import logsumexp


class MBAR(mixture):
    """A class for Multistate Bennett Acceptance Ratio amples (MICS)

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
    def __init__(self, samples, title="Untitled", verbose=False, tol=1.0E-12):

        m, n, neff = self.__define__(samples, title, verbose)

        mb = self.MBAR = mbar.MBAR(np.hstack(self.u), n, relative_tolerance=tol,
                                   initial_f_k=self.f)

        self.f = mb.f_k
        verbose and info("Free energies after convergence:", self.f)

        flnpi = (self.f + np.log(n/sum(n)))[:, np.newaxis]
        self.u0 = [-logsumexp(flnpi - u) for u in self.u]
        self.P = [np.exp(flnpi - self.u[i] + self.u0[i]) for i in range(m)]

        Theta = mb._computeAsymptoticCovarianceMatrix(np.exp(mb.Log_W_nk), mb.N_k)
        self.Theta = np.array(Theta)
        verbose and info("Free-energy covariance matrix:", self.Theta)

        self.Overlap = mb.N_k*np.matmul(mb.W_nk.T, mb.W_nk)
        verbose and info("Overlap matrix:", self.Overlap)

    # ======================================================================================
    def __reweight__(self, u, y, ref=0):
        u_ln = np.stack([np.hstack(u).flatten(),                 # new state = 0
                         np.hstack(x[ref, :] for x in self.u)])  # reference state = 1

        A_n = np.hstack(y)  # properties
        n = A_n.shape[0]    # number of properties

        # Compute properties [0:n-1] at state 0 and property 0 at state 1:
        smap = np.arange(2) if n == 0 else np.block([[np.zeros(n, np.int), 1],  # states
                                                     [np.arange(n), 0]])        # properties

        results = self.MBAR.computeExpectationsInner(A_n, u_ln, smap, return_theta=True)

        # Covariance matrix of x = log(c), whose size is 2*(n+1) x 2*(n+1):
        Theta = results['Theta']

        if n == 0:
            fu = results['free energies'][0] - results['free energies'][1]
            d2fu = Theta[0, 0] + Theta[1, 1] - 2*Theta[0, 1]
            return np.array([fu]), np.array([[d2fu]])

        # Functions, whose number is n+1:
        fu = np.array([results['free energies'][0] - results['free energies'][n]])
        yu = results['observables'][0:n]

        # Gradient:
        #     fu = -ln(c[n+1]/c[2*n+1]) = x[2*n+1] - x[n+1]
        #     yu[i] = c[i]/c[n+1+i] = exp(x[i] - x[n+1+i])
        G = np.zeros([2*(n+1), n+1])
        G[n+1, 0] = -1.0
        G[2*n+1, 0] = 1.0
        delta = yu - results['Amin'][0:n]
        for i in range(n):
            G[i, i+1] = delta[i]
            G[n+1+i, i+1] = -delta[i]

        return np.concatenate([fu, yu]), multi_dot([G.T, Theta, G])
