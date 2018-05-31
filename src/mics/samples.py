"""
.. module:: samples
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`sample`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

# TODO: save potential and autocor as strings rather than lambda functions, so that
#       one can use pickle to save a sample or a mixture object.

from copy import deepcopy

import numpy as np
from pymbar import timeseries

from mics.funcs import func
from mics.utils import covariance
from mics.utils import info
from mics.utils import multimap


class sample:
    """
    A sample of configurations collected at a specific equilibrium state whose
    probability density function is proportional to `exp(-potential(x))`. Each
    configuration `x` is represented by a set of collective variables.

    Parameters
    ----------
        dataset : pandas.DataFrame
            A data frame whose rows contain the values of a set of collective
            variables representing configurations sampled according to a given
            probability distribution.
        potential : string
            The reduced potential that defines the probability density at the
            sampled equilibrium sample. This must be a function of the column
            names in `dataset`.
        autocorr : string, optional, default = potential
            An property to be used for autocorrelation analysis and effective
            sample size calculation through the Overlapping Batch Mean (OBM)
            method. This must be a function of the column names in `dataset`.
        batchsize : int, optional, default = sqrt(dataset size)
            The size of each batch (window) to be used for the OBM analysis.
        **constants : keyword arguments
            Value assignment for constants present in functions `potential`
            and `autocorr`.

    """

    def __init__(self, dataset, potential, autocorr=None, batchsize=None, verbose=False, **constants):

        names = list(dataset.columns)
        n = len(dataset)
        b = self.b = batchsize if batchsize else int(np.sqrt(n))

        if verbose:
            info("\n=== Setting up new sample ===")
            info("Properties:", ", ".join(names))
            info("Constants:", constants)
            info("Reduced potential:", potential)
            info("Autocorrelation analysis property:", autocorr if autocorr else potential)
            info("Sample size:", n)
            info("Batch size:", b)

        self.dataset = dataset
        self.potential = func(potential, names, constants)
        self.autocorr = self.potential if autocorr is None else func(autocorr, names, constants)
        y = multimap([self.autocorr.lambdify()], dataset)
        ym = np.mean(y, axis=1)
        S1 = covariance(y, ym, 1).item(0)
        Sb = covariance(y, ym, b).item(0)
        if not (np.isfinite(S1) and np.isfinite(Sb)):
            raise FloatingPointError("unable to determine effective sample size")
        self.neff = n*S1/Sb

        if verbose:
            info("Variance disregarding autocorrelation:", S1)
            info("Variance via Overlapping Batch Means:", Sb)
            info("Effective sample size:", self.neff)


class pooledSample:
    """
    A pool of independently collected samples.

    """

    # ======================================================================================
    def __init__(self, label="", verbose=False):
        self.samples = list()
        self.label = str(label)
        self.verbose = verbose

    # ======================================================================================
    def add(self, *args, **kwargs):
        self.samples.append(sample(*args, verbose=self.verbose, **kwargs))

    # ======================================================================================
    def copy(self):
        return deepcopy(self)

    # ======================================================================================
    def subsample(self, compute_inefficiency=True):
        self.verbose and info("Performing subsampling...")
        for (i, sample) in enumerate(self.samples):
            n = len(sample.dataset)
            self.verbose and info("Original sample size:", n)
            old = sample.dataset.index
            if compute_inefficiency:
                y = multimap([sample.autocorr.lambdify()], sample.dataset)
                g = timeseries.statisticalInefficiency(y[0])
                self.verbose and info("Statistical inefficency via integrated ACF:", g)
            else:
                g = n/sample.neff
                self.verbose and info("Statistical inefficency via Overlapping Batch Means:", g)
            new = timeseries.subsampleCorrelatedData(old, g)
            sample.dataset = sample.dataset.reindex(new)
            sample.neff = len(new)
            self.verbose and info("New sample size:", sample.neff)
        return self

    # ======================================================================================
    def __getitem__(self, i):
        return self.samples[i]

    # ======================================================================================
    def __len__(self):
        return len(self.samples)
