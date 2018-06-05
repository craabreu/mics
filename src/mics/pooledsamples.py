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
    A list with extensions for dealing with collections of :class:`sample`
    objects. From a `pooledsample`, one can call :func:`~sample.subsampling`
    and :func:`~sample.averaging` for all samples simultaneously. One can
    also generate a :class:`mixture` object for multistate analysis with
    either :class:`MICS` or :class:`MBAR`.

    """

    def __iadd__(self, item):
        if isinstance(item, mics.sample):
            self.append(item)
        else:
            super(pooledsample, self).__iadd__(item)
        return self

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
