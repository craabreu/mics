"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np

from mics.utils import info
from mics.utils import multimap
from mics.utils import overlapSampling


class Mixture:
    """A mixture of independently collected samples (MICS)

        Args:
            states (list or tuple):
                a list of states.
            title (str, optional):
                a title.
            verbose (bool, optional):
                a verbosity tag.
            tol (float, optional):
                a tolerance.

    """

    def __init__(self, states, title="Untitled", verbose=False, tol=1.0E-8):

        info(verbose, "Setting up MICS case:", title)

        m = self.m = len(states)
        info(verbose, "Number of states:", m)

        if m == 0:
            raise ValueError("state set is empty")

        def names(i):
            return list(states[i].sample.columns.values)
        properties = names(0)
        info(verbose, "Properties:", ", ".join(properties))

        if any([names(i) != properties for i in range(m)]):
            raise ValueError("inconsistent data")

        n = self.n = [states[i].sample.shape[0] for i in range(m)]
        info(verbose, "Sample sizes:", n)

        neff = np.array([states[i].neff for i in range(m)])
        info(verbose, "Effective sample sizes:", neff)

        pi = self.pi = neff/sum(neff)
        info(verbose, "Mixture composition:", pi)

        potentials = [states[i].potential for i in range(m)]
        u = [multimap(potentials, states[i].sample) for i in range(m)]

        f = self.f = overlapSampling(u)
        info(verbose, "Initial free-energy guess:", f)
