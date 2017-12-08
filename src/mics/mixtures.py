"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

from copy import deepcopy

import numpy as np
import pandas as pd
from numpy.linalg import multi_dot

from mics.samples import pool
from mics.utils import InputError
from mics.utils import cases
from mics.utils import genfunc
from mics.utils import info
from mics.utils import jacobian
from mics.utils import multimap
from mics.utils import overlapSampling


class mixture:
    """A mixture of independently collected samples (MICS)

        Args:
            samples (list or tuple):
                a list of samples.
            title (str, optional):
                a title.
            verbose (bool, optional):
                a verbosity tag.
            tol (float, optional):
                a tolerance.

    """

    # ======================================================================================
    def __define__(self, samples, title, verbose):

        np.set_printoptions(precision=4, threshold=15, edgeitems=4, suppress=True)

        self.title = title
        self.method = type(self).__name__
        verbose and info("Setting up %s case:" % self.method, title)

        m = self.m = len(samples)
        if m == 0:
            raise InputError("list of samples is empty")
        verbose and info("Number of samples:", m)

        if type(samples) is pool:
            self.samples = deepcopy(samples.samples)
            self.label = samples.label
        else:
            self.samples = deepcopy(samples)
            self.label = ""

        names = self.names = list(samples[0].dataset.columns)
        if any(list(s.dataset.columns) != names for s in samples):
            raise InputError("provided samples have distinct properties")
        verbose and info("Properties:", ", ".join(names))

        n = self.n = np.array([s.dataset.shape[0] for s in samples])
        verbose and info("Sample sizes:", str(self.n))

        neff = self.neff = np.array([s.neff for s in samples])
        verbose and info("Effective sample sizes:", neff)

        potentials = [s.potential for s in samples]
        self.u = [multimap(potentials, s.dataset) for s in samples]

        self.f = overlapSampling(self.u)
        verbose and info("Initial free-energy guess:", self.f)

        self.frame = pd.DataFrame(index=np.arange(m) + 1)
        if self.label:
            self.states = ["%s = %s" % (self.label, s.label) for s in samples]
            self.frame[self.label] = [s.label for s in samples]
        else:
            self.states = ["state %d" % (i+1) for i in range(m)]

        return m, n, neff

    # ======================================================================================
    def compute(self, functions, constants):
        if isinstance(functions, str):
            funcs = [genfunc(functions, self.names, constants)]
        else:
            funcs = [genfunc(f, self.names, constants) for f in functions]
        return [multimap(funcs, s.dataset) for s in self.samples]

    # ======================================================================================
    def free_energies(self, reference=0):
        """
        Returns a data frame containing the relative free energies of the datasetd samples
        of a `mixture`, as well as their standard errors.

        """
        frame = self.frame.copy()
        frame["f"] = self.f - self.f[reference]
        T = self.Theta
        frame["df"] = np.sqrt(np.diag(T) - 2*T[:, reference] + T[reference, reference])
        return frame

    # ======================================================================================
    def reweighting(self,
                    potential,
                    properties,
                    combinations={},
                    conditions=pd.DataFrame(),
                    verbose=False,
                    **kwargs):
        """
        Performs reweighting of the properties computed by `functions` from the mixture to
        the samples determined by the provided `potential` with all `parameter` values.

        Args:
            potential (function/string):
            properties (dict of functions/strings):
            combinations (dict of strings):
            parameter (pandas.DataFrame):
            **kwargs:

        """
        verbose and info("Reweighting requested - %s case:" % self.method, self.title)
        verbose and info("Reduced potential:", potential)

        try:
            y = self.compute(properties.values(), kwargs)
            properties_needed = False
        except (InputError, KeyError):
            properties_needed = True

        names = list(properties.keys())
        if (combinations):
            try:
                func, Jac = jacobian(combinations.values(), names, kwargs)
                jacobian_needed = False
            except InputError:
                jacobian_needed = True

        results = list()
        for constants in cases(potential, conditions, kwargs, verbose):
            u = self.compute(potential, constants)
            if properties_needed:
                y = self.compute(properties.values(), constants)
            yu, Theta = self.__reweight__(u, y)
            dyu = np.sqrt(np.diagonal(Theta))

            if (combinations):
                if jacobian_needed:
                    func, Jac = jacobian(combinations.values(), names, constants)
                g = func(yu).flatten()
                J = Jac(yu)
                dg = np.sqrt(np.diagonal(multi_dot([J, Theta, J.T])))
                results.append(np.block([[yu, g], [dyu, dg]]).T.flatten())
            else:
                results.append(np.stack([yu, dyu]).T.flatten())

        header = sum([[p, "d_"+p] for p in names + list(combinations.keys())], [])
        return pd.concat([conditions, pd.DataFrame(results, columns=header)], axis=1)

    # ======================================================================================
    def fep(self,
            potential,
            conditions=pd.DataFrame(),
            reference=0,
            verbose=False,
            **kwargs):
        """
        Performs free energy perturbation.

        Args:
            potential (function/string):
            conditions (pandas.DataFrame):
            **kwargs:

        """
        verbose and info("FEP requested - %s case:" % self.method, self.title)
        verbose and info("Reduced potential:", potential)
        results = list()
        for constants in cases(potential, conditions, kwargs, verbose):
            u = self.compute(potential, constants)
            results.append(self.__perturb__(u, reference))
        frame = pd.DataFrame(results, columns=["f", "d_f"])
        return pd.concat([conditions, frame], axis=1)

    # ======================================================================================
    def pmf(self,
            potential,
            property,
            bins=10,
            interval=None,
            verbose=False,
            **kwargs):

        verbose and info("PMF requested - %s case:" % self.method, self.title)
        verbose and info("Reduced potential:", potential)
        u = self.compute(potential, kwargs)

        z = self.compute(property, kwargs)
        if interval:
            (zmin, zmax) = interval
        else:
            zmin = min(np.amin(x[0, :]) for x in z)
            zmax = max(np.amax(x[0, :]) for x in z)
        delta = (zmax - zmin)/bins
        ibin = [np.floor((x[0:1, :] - zmin)/delta).astype(int) for x in z]

        results = list()
        for i in range(bins):
            zc = zmin + delta*(i + 0.5)
            verbose and info("Bin[%d]:" % (i + 1), "%s = %s" % (property, str(zc)))
            y = [np.equal(x, i).astype(np.float) for x in ibin]
            yu, Theta = self.__reweight__(u, y)
            dyu = np.sqrt(Theta)
            results.append([zc, -np.log(yu[0]), dyu[0, 0]/yu[0]])

        return pd.DataFrame(results, columns=[property, "pmf", "d_pmf"])

    # ======================================================================================
    def histograms(self, property="u0", bins=100, **kwargs):
        y = self.u0 if property == "u0" else self.compute(property, kwargs)
        ymin = min([np.amin(x) for x in y])
        ymax = max([np.amax(x) for x in y])
        delta = (ymax - ymin)/bins
        center = [ymin + delta*(i + 0.5) for i in range(bins)]
        frame = pd.DataFrame({property: center})
        for (i, s) in enumerate(self.states):
            frame[s] = np.histogram(y[i], bins, (ymin, ymax))[0]
        return frame
