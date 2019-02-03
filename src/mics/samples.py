"""
.. module:: samples
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`sample`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import numpy as np
from pymbar import timeseries

import mics
from mics.funcs import deltaMethod
from mics.funcs import func
from mics.utils import covariance
from mics.utils import info
from mics.utils import multimap
from mics.utils import propertyDict
from mics.utils import stdError


class sample:
    """
    A sample of configurations distributed according to a :term:`PDF`
    proportional to ``exp(-u(x))``. Each configuration ``x`` is represented
    by a set of collective variables from which one can evaluate the reduced
    potential ``u(x)``, as well as other properties of interest.

    Parameters
    ----------
        dataset : pandas.DataFrame
            A data frame whose column names are collective variables used to
            represent the sampled comfigurations. The rows must contain a time
            series of these variables, obtained by simulating the system at a
            state with known reduced potential.
        potential : str
            A mathematical expression defining the reduced potential of the
            simulated state. This must be a function of the column names in
            `dataset` and can also depend on external parameters passed as
            keyword arguments (see below).
        acfun : str, optional, default=potential
            A mathematical expression defining a property to be used for
            :term:`OBM` autocorrelation analysis and effective sample size
            calculation. It must be a function of the column names in `dataset`
            and can also depend on external parameters passed as keyword
            arguments (see below).
        batchsize : int, optional, default=sqrt(len(dataset))
            The size of each batch (window) to be used in the :term:`OBM`
            analysis. If omitted, then the batch size will be the integer
            part of the square root of the sample size.
        **constants : keyword arguments
            A set of keyword arguments passed as name=value, aimed to define
            external parameter values for the evaluation of the mathematical
            expressions in `potential` and `acfun`. They can also be used as
            labels to distinguish samples from each other, in this case not
            necessary being present in the mentioned expressions.

    """

    def __init__(self, dataset, potential, acfun=None, batchsize=None, **constants):
        names = dataset.columns.tolist()
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

    def __add__(self, other):
        return mics.pooledsample([self]) + other

    def __radd__(self, other):
        return mics.pooledsample(other) + self

    def subsampling(self, integratedACF=True):
        """
        Performs inline subsampling based on the statistical inefficiency ``g``
        of the specified attribute `acfun` of :class:`sample`, aiming at
        obtaining a sample of :term:`IID` configurations. Subsampling is done
        via jumps of varying sizes around ``g``, so that the sample size decays
        by a factor of approximately ``1/g``.

        Parameters
        ----------
            integratedACF : bool, optional, default=True
                If true, the integrated :term:`ACF` method :cite:`Chodera_2007`
                will be used for computing the statistical inefficiency.
                Otherwise, the :term:`OBM` method will be used instead.

        Returns
        -------
            :class:`sample`
                Although the subsampling is done inline, the new sample is
                returned for chaining purposes.

        """
        n = len(self.dataset)
        if mics.verbose:
            info("\n=== Subsampling via %s ===" % ("integrated ACF" if integratedACF else "OBM"))
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
            info("Statistical inefficiency:", g)
            info("New sample size:", self.neff)
        return self

    def averaging(self, properties, combinations={}, **constants):
        """
        Computes averages and uncertainties of configurational properties. In
        addition, computes combinations among these averages while automatically
        handling uncertainty propagation.

        Parameters
        ----------
            properties : dict(str: str)
                A dictionary associating names to mathematical expressions. This
                is used to define functions of the collective variables included
                in the samples. Then, averages of these functions will be
                evaluated at all sampled states, along with their uncertainties.
                The expressions might also depend on parameters passed as
                keyword arguments (see below).
            combinations : dict(str: str), optional, default={}
                A dictionary associating names to mathematical expressions. This
                is used to define functions of the names passed as keys in the
                `properties` dictionary. The expressions might also depend on
                parameters passed as keyword arguments (see below).
            **constants : optional keyword arguments
                A set of arguments passed as ``name=value``, used to define
                parameter values for evaluating the mathematical expressions
                in both `properties` and `combinations`.

        Returns
        -------
            pandas.DataFrame
                A data frame containing the computed averages and combinations,
                as well as their estimated standard errors.

        """
        variables = self.dataset.columns.tolist()
        functions = [func(f, variables, constants).lambdify() for f in properties.values()]
        y = multimap(functions, self.dataset)
        ym = np.mean(y, axis=1)
        Theta = covariance(y, ym, self.b)
        result = propertyDict(properties.keys(), ym, stdError(Theta))
        if combinations:
            delta = deltaMethod(combinations.values(), properties.keys(), constants)
            (h, dh) = delta.evaluate(ym, Theta)
            result.update(propertyDict(combinations.keys(), h, dh))
        return result.to_frame(0)
