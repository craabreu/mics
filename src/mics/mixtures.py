"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
import pandas as pd
from numpy.linalg import multi_dot

from mics.samples import pool
from mics.utils import InputError
from mics.utils import cases
from mics.utils import derivative
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
        self.verbose = verbose
        self.method = type(self).__name__
        verbose and info("Setting up %s case:" % self.method, title)

        m = self.m = len(samples)
        if m == 0:
            raise InputError("list of samples is empty")
        verbose and info("Number of samples:", m)

        if type(samples) is pool:
            self.samples = samples.samples
            self.label = samples.label
        else:
            self.samples = samples
            self.label = ""

        names = self.names = list(samples[0].dataset.columns)
        if any(list(s.dataset.columns) != names for s in samples):
            raise InputError("provided samples have distinct properties")
        verbose and info("Properties:", ", ".join(names))

        n = self.n = np.array([s.dataset.shape[0] for s in samples])
        verbose and info("Sample sizes:", str(self.n))

        potentials = [s.potential for s in samples]
        self.u = [multimap(potentials, s.dataset) for s in samples]

        self.f = overlapSampling(self.u)
        verbose and info("Initial free-energy guess:", self.f)

        neff = self.neff = np.array([s.neff for s in samples])

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
                    properties={},
                    derivatives={},
                    combinations={},
                    conditions=pd.DataFrame(),
                    reference=0,
                    **kwargs):
        """
        Performs reweighting of the properties computed by `functions` from the mixture to
        the samples determined by the provided `potential` with all `parameter` values.

        Args:
            potential (string):
            properties (dict of strings):
            combinations (dict of strings):
            derivatives (dict of tuples):
            conditions (pandas.DataFrame):
            verbose (boolean):
            **kwargs:

        """
        # TODO: look for duplicated names or reserved keyword 'f' in properties
        # TODO: look for duplicated names in properties, derivatives, and combinations
        # TODO: allow limited recursion in combinations

        names = ['f'] + list(properties.keys())

        if self.verbose:
            info("Reweighting requested in %s case:" % self.method, self.title)
            info("Reduced potential:", potential)
            kwargs and info("Provided constants: ", kwargs)

        if not derivatives:

            try:
                y = self.compute(properties.values(), kwargs)
                properties_needed = False
            except (InputError, KeyError):
                properties_needed = True

            if (combinations):
                try:
                    func, Jac = jacobian(combinations.values(), names, kwargs)
                    jacobian_needed = False
                except InputError:
                    jacobian_needed = True

            results = list()
            for constants in cases(conditions, kwargs, self.verbose):
                u = self.compute(potential, constants)
                if properties_needed:
                    y = self.compute(properties.values(), constants)
                g, Theta = self.__reweight__(u, y, reference)
                dg = np.sqrt(np.diagonal(Theta))

                if (combinations):
                    if jacobian_needed:
                        func, Jac = jacobian(combinations.values(), names, constants)
                    h = func(g).flatten()
                    J = Jac(g)
                    dh = np.sqrt(np.diagonal(multi_dot([J, Theta, J.T])))
                    results.append(np.block([[g, h], [dg, dh]]).T.flatten())
                else:
                    results.append(np.stack([g, dg]).T.flatten())

            header = sum([[x, "d"+x] for x in names + list(combinations.keys())], [])
#             if conditions.empty:
#                 return dict(zip(header, results[0]))
#             else:
#                 return pd.concat([conditions, pd.DataFrame(results, columns=header)], 1)
            return pd.concat([conditions, pd.DataFrame(results, columns=header)], 1)

        else:

            def dec(x):
                return "__%s__" % x

            parameters = list(conditions.columns) + list(kwargs.keys())
            zyx = [(key, value[0], value[1]) for key, value in derivatives.items()]

            props = {}
            combs = {}
            for x in set(x for z, y, x in zyx):
                props[dec(x)] = derivative(potential, x, parameters)

            for z, y, x in zyx:
                if y == 'f':
                    combs[z] = dec(x)
                else:
                    dydx = derivative(properties[y], x, parameters)
                    props[dec(z)] = "%s - (%s)*(%s)" % (dydx, props[dec(x)], properties[y])
                    combs[z] = "%s + (%s)*(%s)" % (dec(z), dec(x), y)

            unwanted = sum([[x, "d"+x] for x in props.keys()], [])

            return self.reweighting(potential, dict(properties, **props), {},
                                    dict(combs, **combinations), conditions, reference,
                                    **kwargs).drop(unwanted, axis=1)

    # ======================================================================================
    def pmf(self,
            potential,
            property,
            bins=10,
            interval=None,
            **kwargs):

        self.verbose and info("PMF requested - %s case:" % self.method, self.title)
        self.verbose and info("Reduced potential:", potential)
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
            self.verbose and info("Bin[%d]:" % (i + 1), "%s = %s" % (property, str(zc)))
            y = [np.equal(x, i).astype(np.float) for x in ibin]
            yu, Theta = self.__reweight__(u, y)
            if yu[1] > 0.0:
                dyu = np.sqrt(Theta[1, 1])
                print([zc, -np.log(yu[1]), dyu/yu[1]])
                results.append([zc, -np.log(yu[1]), dyu/yu[1]])

        return pd.DataFrame(results, columns=[property, "pmf", "d_pmf"])

    # ======================================================================================
    def histograms(self, property="u0", bins=100, **kwargs):
        if property == "u0":
            y = self.u0
        elif property == "state":
            w = np.arange(self.m) + 1
            wsum = sum(w)
            y = [wsum*np.average(p, axis=0, weights=w) for p in self.P]
        elif property == "potential":
            y = [self.u[i][i, :] for i in range(self.m)]
        else:
            y = self.compute(property, kwargs)
        ymin = min([np.amin(x) for x in y])
        ymax = max([np.amax(x) for x in y])
        delta = (ymax - ymin)/bins
        center = [ymin + delta*(i + 0.5) for i in range(bins)]
        frame = pd.DataFrame({property: center})
        for (i, s) in enumerate(self.states):
            frame[s] = np.histogram(y[i], bins, (ymin, ymax))[0]
        return frame
