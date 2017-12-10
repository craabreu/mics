"""
.. module:: MBAR
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`MBAR`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
from numpy.linalg import multi_dot
from pymbar import mbar
from pymbar import timeseries

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
    def __init__(self, samples, title="Untitled", verbose=False, tol=1.0E-12,
                 subsample=True, compute_acf=True):

        m, n, neff = self.__define__(samples, title, verbose)

        if subsample:
            if compute_acf:
                verbose and info("Subsampling method:", "integrated ACF")
            else:
                verbose and info("Subsampling method:", "overlapping batch means")
            for (i, s) in enumerate(self.samples):
                old = s.dataset.index
                if compute_acf:
                    new = timeseries.subsampleCorrelatedData(s.autocorr(s.dataset))
                else:
                    new = timeseries.subsampleCorrelatedData(old, g=n[i]/neff[i])
                s.dataset = s.dataset.reindex(new)
                self.u[i] = self.u[i][:, new]
                n[i] = len(new)
                verbose and info("Size of subsampled dataset %d:" % (i + 1), n[i])
        else:
            verbose and info("Subsampling method:", "none")

        g = (self.f + np.log(n/sum(n)))[:, np.newaxis]
        self.u0 = [-logsumexp(g - x) for x in self.u]

        mb = self.MBAR = mbar.MBAR(np.hstack(self.u), n, relative_tolerance=tol,
                                   initial_f_k=self.f)

        self.f = mb.f_k
        verbose and info("Free energies after convergence:", self.f)

        Theta = mb._computeAsymptoticCovarianceMatrix(np.exp(mb.Log_W_nk), mb.N_k)
        self.Theta = np.array(Theta)
        verbose and info("Free-energy covariance matrix:", self.Theta)

        self.Overlap = mb.N_k*np.matmul(mb.W_nk.T, mb.W_nk)
        verbose and info("Overlap matrix:", self.Overlap)

    # ======================================================================================
    def __reweight__(self, u, y, ref=0):
        A_n = np.hstack(y)  # properties
        n = A_n.shape[0]    # number of properties
        u_ln = np.stack([np.hstack(u).flatten(),                 # new state = 0
                         np.hstack(x[ref, :] for x in self.u)])  # reference state = 1

        # Compute properties [0:n-1] at state 0 and property 0 at state 1:
        smap = np.block([[np.zeros(n, np.int), 1],  # states
                         [np.arange(n), 0]])        # properties
        results = self.MBAR.computeExpectationsInner(A_n, u_ln, smap, return_theta=True)

        # Functions, whose number is n+1:
        fu = [results['free energies'][0] - results['free energies'][n]]
        yu = results['observables'][0:n]

        # Covariance matrix of x = log(c), whose size is 2*(n+1) x 2*(n+1):
        Theta = results['Theta']

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

    # ======================================================================================
    def __perturb__(self, u, ref=0):
        uself = np.hstack(self.u)
        u_ln = np.stack([uself[ref, :], np.hstack(u).flatten()])
        f, df = self.MBAR.computePerturbedFreeEnergies(u_ln, compute_uncertainty=True)
        return f[0, 1], df[0, 1]
