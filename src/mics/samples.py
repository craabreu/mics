"""
.. module:: samples
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`sample`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

# TODO: save potential and autocor as strings rather than lambda functions, so that
#       one can use pickle to save a sample or a mixture object.

import numpy as np
from pymbar import timeseries

import mics
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
        acfun : string, optional, default = potential
            An property to be used for autocorrelation analysis and effective
            sample size calculation through the Overlapping Batch Mean (OBM)
            method. This must be a function of the column names in `dataset`.
        batchsize : int, optional, default = sqrt(dataset size)
            The size of each batch (window) to be used for the OBM analysis.
        **constants : keyword arguments
            Value assignment for constants present in functions `potential`
            and `autocorr`.

    """

    def __init__(self, dataset, potential, acfun=None, batchsize=None, **constants):
        names = list(dataset.columns)
        n = len(dataset)
        b = self.b = batchsize if batchsize else int(np.sqrt(n))

        if mics.verbose:
            info("\n=== Setting up new sample ===")
            info("Properties:", ", ".join(names))
            info("Constants:", constants)
            info("Reduced potential function:", potential)
            info("Autocorrelation analysis function:", acfun if acfun else potential)
            info("Sample size:", n)
            info("Batch size:", b)

        self.dataset = dataset
        self.potential = func(potential, names, constants)
        self.acfun = self.potential if acfun is None else func(acfun, names, constants)
        y = multimap([self.acfun.lambdify()], dataset)
        ym = np.mean(y, axis=1)
        S1 = covariance(y, ym, 1).item(0)
        Sb = covariance(y, ym, b).item(0)
        if not (np.isfinite(S1) and np.isfinite(Sb)):
            raise FloatingPointError("unable to determine effective sample size")
        self.neff = n*S1/Sb

        if mics.verbose:
            info("Variance disregarding autocorrelation:", S1)
            info("Variance via Overlapping Batch Means:", Sb)
            info("Effective sample size:", self.neff)

    def subsample(self, integratedACF=True):
        """
        Performs inline subsampling based on the statistical inefficency `g`
        of the specified function `acfun`. The jumps are not uniformly sized,
        but vary around `g` so that the sample size decays by a factor of
        approximately `1/g`.

        Parameters
        ----------
            integratedACF : bool, optional, default = True
                If true, the integrated autocorrelation function method will
                be used for computing the statistical inefficency. Otherwise,
                the Overlapping Batch Mean (OBM) method will be used instead.

        Returns
        -------
            mics.sample
                Although the subsampling is done in line, the new sample is
                returned for chaining purposes.

        """

        n = len(self.dataset)
        if mics.verbose:
            info("\n=== Subsampling via %s ===" % "integrated ACF" if integratedACF else "OBM")
            info("Original sample size:", n)
        if integratedACF:
            y = multimap([self.acfun.lambdify()], self.dataset)
            g = timeseries.statisticalInefficiency(y[0])
        else:
            g = n/self.neff
        new = timeseries.subsampleCorrelatedData(self.dataset.index, g)
        self.dataset = self.dataset.reindex(new)
        self.neff = len(new)
        if mics.verbose:
            info("Statistical inefficency:", g)
            info("New sample size:", self.neff)
        return self


class pooledsample(list):
    """
    A python list subclass with extensions for dealing with collections of
    `mics.sample` objects.

    """

    def __iadd__(self, item):
        if isinstance(item, sample):
            self.append(item)
        else:
            super(pooledsample, self).__iadd__(item)
        return self

    def subsample(self, integratedACF=True):
        """
        Performs inline subsampling of all samples in the list.

        Parameters
        ----------
            integratedACF : bool, optional, default = True
                If true, the integrated autocorrelation function method will
                be used for computing the statistical inefficency. Otherwise,
                the Overlapping Batch Mean (OBM) method will be used instead.

        Returns
        -------
            mics.pooledsample
                Although the subsampling is done in line, the new pooled sample
                is returned for chaining purposes.

        """
        for sample in self:
            sample.subsample(integratedACF)
        return self

    def mixture(self, method=mics.MICS()):
        return mics.mixture(self, method)
