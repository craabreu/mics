"""
.. module:: samples
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`pooledsample`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import numpy as np
import pandas as pd

import mics
from mics.funcs import qualifiers


class pooledsample(list):
    """
    A python list, but with extensions for dealing with collections of
    :class:`sample` objects. For instance, from a `pooledsample` one can call
    :func:`~sample.subsampling` and :func:`~sample.averaging` for all samples
    simultaneously. One can also create a :class:`mixture` object for
    multistate analysis with either :class:`MICS` or :class:`MBAR`.

    """

    def __init__(self, iterable=0):
        # This constructor uses __iadd__ to ensure that only :class:`sample`
        # objects are accepted.
        super(pooledsample, self).__init__()
        if iterable is not 0:
            self.__iadd__(iterable)

    def __iadd__(self, other):
        # This makes the += operator act like a chained `append` method, but
        # accepting only :class:`sample` objects as arguments.
        if isinstance(other, mics.sample):
            self.append(other)
        elif hasattr(other, "__iter__"):
            for item in other:
                self.__iadd__(item)
        else:
            raise ValueError("A pooledsample can only contain sample objects")
        return self

    def __add__(self, other):
        return pooledsample(super(pooledsample, self).__add__(pooledsample(other)))

    def __getitem__(self, key):
        # This is necessary for slices to be returned as pooledsample objects.
        item = super(pooledsample, self).__getitem__(key)
        return item if isinstance(item, mics.sample) else pooledsample(item)

    def __qualifiers__(self):
        functions = [sample.potential for sample in self]
        return pd.DataFrame(index=np.arange(len(self)), data=qualifiers(functions))

    def averaging(self, properties, combinations={}, **constants):
        """
        For all :class:`sample` objects in the list, performs :func:`sample.averaging` and
        uncertainty analysis of specified properties. Combinations among
        averages can also be computed, with uncertainty propagation being handled automatically.

        Parameters
        ----------
            properties : dict(str: str)
                A dictionary associating names to mathematical expressions, thus
                defining a set of properties whose averages must be evaluated at
                the sampled states. The expressions might depend on the sample's
                collective variables, as well as on parameters passed as keyword
                arguments.
            combinations : dict(str: str), optional, default={}
                A dictionary associating names to mathematical expressions, thus
                defining combinations among average properties at the sampled
                state. The expressions might depend on the names (keys) defined
                in `properties`, as well as on external parameters passed as
                keyword arguments.
            **constants : keyword arguments
                A set of keyword arguments passed as ``name=value``, aimed to
                defining external parameter values for the evaluation of
                mathematical expressions.

        Returns
        -------
            pandas.DataFrame
                A data frame containing the computed averages and combinations,
                as well as their estimated standard errors.

        """
        results = list()
        for (index, sample) in enumerate(self):
            results.append(sample.averaging(properties, combinations, **constants))
        return self.__qualifiers__().join(pd.concat(results, ignore_index=True))

    def mixture(self, engine):
        """
            Generates a :class:`mixture` object.

        Parameters
        ----------
            engine: :class:`MICS` or :class:`MBAR`

        Returns
        -------
            :class:`mixture`

        """
        return mics.mixture(self, engine)

    def subsampling(self, integratedACF=True):
        """
        Performs inline subsampling of all :class:`sample` objects in the list.

        Parameters
        ----------
            integratedACF : bool, optional, default=True
                If true, the integrated autocorrelation function method will
                be used for computing the statistical inefficency. Otherwise,
                the :term:`OBM` method will be used instead.

        Returns
        -------
            :class:`pooledsample`
                Although the subsampling is done in line, the new pooled sample
                is returned for chaining purposes.

        """
        for sample in self:
            sample.subsampling(integratedACF)
        return self
