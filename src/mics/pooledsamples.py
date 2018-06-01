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
    A python list subclass with extensions for dealing with collections of
    `mics.sample` objects.

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

    def averaging(self,
                  properties,
                  combinations={},
                  **constants):
        """
        Performs averaging of specified properties and uncertainty analysis
        via Overlapping Batch Means. Combinations of these averages can also
        be computed, with uncertainty propagation being automatically handled.

        Parameters
        ----------
            properties : dict(string: string)

            combinations : dict(string: string), optional, default = {}

            **constants : keyword arguments

        Returns
        -------
            pandas.DataFrame
                A data frame containing the computed averages and combinations,
                as well as their computed standard errors.

        """
        results = list()
        for (index, sample) in enumerate(self):
            results.append(sample.averaging(properties, combinations, index, **constants))
        return self.__qualifiers__().join(pd.concat(results))

    def mixture(self, method=mics.MICS()):
        return mics.mixture(self, method)

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
