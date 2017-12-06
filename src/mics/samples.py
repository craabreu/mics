"""
.. module:: samples
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`sample`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np

from mics.utils import covariance
from mics.utils import genfunc
from mics.utils import multimap


class sample:
    """
    A sample of configurations collected at a specific equilibrium state, aimed to be part
    of a mixture of independently collected samples (MICS).

        Args:
            dataset (pandas.DataFrame):
                a data frame whose rows represent configurations datasetd according to a
                given probability distribution and whose columns contain a number of
                properties evaluated for such configurations.
            potential (function):
                the reduced potential that defines the equilibrium sample. This function
                might for instance receive **x** and return the result of an element-wise
                calculation involving **x["a"]**, **x["b"]**, etc, with **"a"**, **"b"**,
                etc being names of properties in **dataset**.
            autocorr (function, optional):
                a function similar to **potential**, but whose result is an autocorrelated
                property to be used for determining the effective dataset size. If omitted,
                **potential** will be used to for this purpose.

        Note:
            Formally, functions **potential** and **autocorr** must receive **x** and
            return **y**, where `length(y) == nrow(x)`.

    """

    def __init__(self, dataset, potential, autocorr=None, batchsize=None, label="", **kwargs):
        names = list(dataset.columns)
        self.dataset = dataset
        self.potential = genfunc(potential, names, kwargs)
        self.label = str(label)
        n = self.n = dataset.shape[0]
        b = self.b = batchsize if batchsize else int(np.sqrt(n))
        self.autocorr = genfunc(autocorr, names, kwargs) if autocorr else self.potential
        y = multimap([self.autocorr], dataset)
        ym = np.mean(y, axis=1)
        S1 = covariance(y, ym, 1).item(0)
        Sb = covariance(y, ym, b).item(0)
        if not (np.isfinite(S1) and np.isfinite(Sb)):
            raise FloatingPointError("unable to determine effective sample size")
        self.neff = n*S1/Sb


class pool:
    """
    A set of independently collected samples.

    """

    # ======================================================================================
    def __init__(self, label=""):
        self.samples = list()
        self.label = str(label)

    # ======================================================================================
    def add(self, *args, **kwargs):
        self.samples.append(sample(*args, **kwargs))

    # ======================================================================================
    def __getitem__(self, i):
        return self.samples[i]

    # ======================================================================================
    def __len__(self):
        return len(self.samples)
