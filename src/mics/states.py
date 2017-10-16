"""
.. module:: states
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`state`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np

from mics.utils import covariance
from mics.utils import multimap


class state:
    """An equilibrium state aimed to be part of a mixture of independently collected
        samples (MICS)

        Args:
            sample (pandas.DataFrame):
                a data frame whose rows represent configurations sampled according to a
                given probability distribution and whose columns contain a number of
                properties evaluated for such configurations.
            potential (function):
                the reduced potential that defines the equilibrium state. This function
                might for instance receive **x** and return the result of an element-wise
                calculation involving **x['a']**, **x['b']**, etc, with **'a'**, **'b'**,
                etc being names of properties in **sample**.
            autocorr (function, optional):
                a function similar to **potential**, but whose result is an autocorrelated
                property to be used for determining the effective sample size. If omitted,
                **potential** will be used to for this purpose.

        Note:
            Formally, functions **potential** and **autocorr** must receive **x** and
            return **y**, where `length(y) == nrow(x)`.

    """

    def __init__(self, sample, potential, autocorr=None):
        self.sample = sample
        self.potential = potential
        n = self.n = sample.shape[0]
        b = self.b = int(round(np.sqrt(n)))
        if autocorr is None:
            y = multimap([potential], sample)
        else:
            y = multimap([autocorr], sample)
        ym = np.mean(y, axis=1)
        S1 = covariance(y, ym, 1).item(0)
        Sb = covariance(y, ym, b).item(0)
        self.neff = int(round(n*S1/Sb))
        if not np.isfinite(self.neff):
            raise FloatingPointError("unable to determine effective sample size")
