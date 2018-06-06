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
    A python list, but with special extensions for dealing with collections of
    :class:`sample` objects. For instance, :func:`~sample.subsampling` and
    :func:`~sample.averaging` can be called for all samples simultaneously.
    There is also a method for creating a :class:`mixture` object directly from
    a pooledsample.

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
        Calls :func:`~sample.averaging` for all samples in the list.

        Parameters
        ----------
            : Same as in :func:`sample.averaging`.

        Returns
        -------
            pandas.DataFrame
                A data frame containing the computed averages and combinations,
                as well as their estimated standard errors, for all samples.

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
        Calls :func:`~sample.subsampling` for all samples in the list.

        Parameters
        ----------
            : Same as in :func:`sample.subsampling`.

        Returns
        -------
            :class:`pooledsample`
                Although the subsampling is done in line, the new pooled sample
                is returned for chaining purposes.

        """
        for sample in self:
            sample.subsampling(integratedACF)
        return self
