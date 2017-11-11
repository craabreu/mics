# -*- coding: utf-8 -*-
"""
.. module:: MBAR
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`MBAR`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
from pymbar import mbar
from pymbar import timeseries

from mics.mixtures import mixture
from mics.utils import info


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
    def __init__(self, samples, title="Untitled", verbose=False, tol=1.0E-8,
                 subsample=False, compute_acf=True):

        m, n, neff = self._definitions(samples, title, verbose)

        if subsample:
            if compute_acf:
                info(verbose, "Subsampling method:", "integrated autocorrelation function")
            else:
                info(verbose, "Subsampling method:", "overlapping batch means")
            for (i, s) in enumerate(self.samples):
                old = s.dataset.index
                if compute_acf:
                    new = timeseries.subsampleCorrelatedData(s.autocorr(s.dataset))
                else:
                    new = timeseries.subsampleCorrelatedData(old, g=n[i]/neff[i])
                s.dataset.drop([k for k in old if k not in set(new)], inplace=True)
                self.u[i] = self.u[i][:, new]
                n[i] = len(new)
            info(verbose, "Subsampled dataset sizes:", str(n))

        self.u = np.hstack(self.u)
        self.MBAR = mbar.MBAR(self.u, n, relative_tolerance=tol, initial_f_k=self.f)

        self.f = self.MBAR.f_k
        info(verbose, "Free energies after convergence:", self.f)

        T = self.MBAR._computeAsymptoticCovarianceMatrix(np.exp(self.MBAR.Log_W_nk),
                                                         self.MBAR.N_k, method='svd-ew')
        self.Theta = np.array(T)
        info(verbose, "Free-energy covariance matrix:", self.Theta)

    # ======================================================================================
    def _reweight(self, u, y):
        A_n = np.hstack(y)
        u_ln = np.hstack(u).flatten()
        n = A_n.shape[0]
        map = np.vstack([np.zeros(n, np.int), np.arange(n)])

        results = self.MBAR.computeExpectationsInner(A_n, u_ln, map, return_theta=True)

        yu = results['observables']
        Q = results['Theta']
        T = Q[0:n, 0:n] + Q[n:2*n, n:2*n] - Q[0:n, n:2*n] - Q[n:2*n, 0:n]
        delta = yu - results['Amin']
        Theta = np.multiply(np.outer(delta, delta), T)
        return yu, Theta
