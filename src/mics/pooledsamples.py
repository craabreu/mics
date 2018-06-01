"""
.. module:: samples
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`pooledsample`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import mics


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
