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

        m, n, b, neff = self._definitions(samples, title, verbose)

        if subsample:
            if compute_acf:
                info(verbose, "Subsampling method:", "Integrated Autocorrelation Function")
            else:
                info(verbose, "Subsampling method:", "Overlapping Batch Means")
            for (i, s) in enumerate(self.samples):
                old = s.dataset.index
                if compute_acf:
                    new = timeseries.subsampleCorrelatedData(s.autocorr(s.dataset))
                else:
                    new = timeseries.subsampleCorrelatedData(old, g=n[i]/neff[i])
                s.dataset[old.isin(new)]
                self.u[i] = self.u[i][:, new]
                n[i] = len(new)
            info(verbose, "Subsampled data:", str(n))

        u_kln = np.zeros([m, m, np.max(n)], np.float64)
        for i in range(m):
            u_kln[i, :, 0:n[i]] = self.u[i]

        self.MBAR = mbar.MBAR(u_kln, n, relative_tolerance=tol, initial_f_k=self.f)

        F, D, Theta = self.MBAR.getFreeEnergyDifferences(return_theta=True)
        self.f = np.array(F[0, :])
        info(verbose, "Free energies after convergence:", self.f)

        self.Theta = np.array(Theta)
        info(verbose, "Free-energy covariance matrix:", self.Theta)

    # ======================================================================================
    def _reweight(self, u, z):
        pass
#         S = range(self.m)
#         pi = self.pi
#         P = self.P
#         pm = self.pm

#         return f, df
