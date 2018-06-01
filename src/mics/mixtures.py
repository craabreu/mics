"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
import pandas as pd
from numpy.linalg import multi_dot

import mics
from mics.funcs import derivative
from mics.funcs import func
from mics.funcs import jacobian
from mics.funcs import qualifiers
from mics.utils import InputError
from mics.utils import cases
from mics.utils import info
from mics.utils import multimap
from mics.utils import overlapSampling
from mics.utils import stdError


class mixture:
    """
    A mixture of independently collected samples (MICS).

    Parameters
    ----------
        samples : list(mics.sample)
            A list of samples.
        method : mics.method, optional, default = mics.MICS()
            A method for mixture-model analysis.

    """

    def __init__(self, samples, method=mics.MICS()):

        self.samples = samples
        self.method = method
        m = self.m = len(samples)
        if mics.verbose:
            # np.set_printoptions(precision=4, threshold=15, edgeitems=4, suppress=True)
            info("\n=== Setting up mixture ===")
            info("Analysis method: ", self.method.__class__.__name__)
            info("Number of samples:", m)

        if m == 0:
            raise InputError("list of samples is empty")

        self.n = np.array([s.dataset.shape[0] for s in samples])
        self.neff = np.array([s.neff for s in samples])
        names = self.names = list(samples[0].dataset.columns)
        if mics.verbose:
            info("Sample sizes:", self.n)
            info("Effective sample sizes:", self.neff)
            info("Properties:", ", ".join(names))

        if any(list(s.dataset.columns) != names for s in samples):
            raise InputError("provided samples have distinct properties")

        functions = [s.potential for s in samples]
        self.frame = pd.DataFrame(index=np.arange(m) + 1, data=qualifiers(functions))

        potentials = [f.lambdify() for f in functions]
        self.u = [multimap(potentials, s.dataset) for s in samples]
        self.f = overlapSampling(self.u)
        mics.verbose and info("Initial free-energy guess:", self.f)

        self.method.__initialize__(self)

    # ======================================================================================
    def __compute__(self, functions, constants):
        if isinstance(functions, str):
            funcs = [func(functions, self.names, constants).lambdify()]
        else:
            funcs = [func(f, self.names, constants).lambdify() for f in functions]
        return [multimap(funcs, s.dataset) for s in self.samples]

    # ======================================================================================
    def free_energies(self, reference=0):
        """
        Computes the free energies of all sampled states relative to a given
        reference state, as well as their standard errors.

        Parameters
        ----------
            reference : int, optional, default = 0
                Specifies which sampled state will be considered as a reference
                for computing free-energy differences.

        Returns
        -------
            pandas.DataFrame
                A data frame containing the free-energy differences and their
                computed standard errors for all sampled states.

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
                    conditions={},
                    reference=0,
                    **kwargs):
        """
        Performs reweighting of the properties computed by `functions` from the
        mixture to the samples determined by the provided `potential` with all
        `parameter` values.

        Parameters
        ----------
            potential : string

            properties : dict(string: string), optional, default = {}

            combinations : dict(string: string), optional, default = {}

            derivatives : dict(string: string), optional, default = {}

            conditions : dict(string: string), optional, default = {}

            reference : int, optional, default = 0

            **kwargs : keyword arguments

        Returns
        -------
            pandas.DataFrame
                A data frame containing the computed quantities at all
                all specified conditions.

        """
        # TODO: look for duplicated names or reserved keyword 'f' in properties
        # TODO: look for duplicated names in properties, derivatives, and combinations
        # TODO: allow limited recursion in combinations

        names = ['f'] + list(properties.keys())
        if isinstance(conditions, dict):
            condframe = pd.DataFrame(data=conditions)
        else:
            condframe = conditions

        if mics.verbose:
            info("\n=== Performing reweighting with %s ===" % self.method.__class__.__name__)
            info("Reduced potential:", potential)
            kwargs and info("Provided constants: ", kwargs)

        if not derivatives:

            try:
                y = self.__compute__(properties.values(), kwargs)
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
            for constants in cases(condframe, kwargs, mics.verbose):
                u = self.__compute__(potential, constants)
                if properties_needed:
                    y = self.__compute__(properties.values(), constants)
                g, Theta = self.method.__reweight__(self, u, y, reference)
                dg = stdError(Theta)

                if (combinations):
                    if jacobian_needed:
                        func, Jac = jacobian(combinations.values(), names, constants)
                    h = func(g).flatten()
                    J = Jac(g)
                    dh = stdError(multi_dot([J, Theta, J.T]))
                    results.append(np.block([[g, h], [dg, dh]]).T.flatten())
                else:
                    results.append(np.stack([g, dg]).T.flatten())

            header = sum([[x, "d"+x] for x in names + list(combinations.keys())], [])
            # if condframe.empty:
            #     return dict(zip(header, results[0]))
            # else:
            #     return pd.concat([condframe, pd.DataFrame(results, columns=header)], 1)
            return pd.concat([condframe, pd.DataFrame(results, columns=header)], 1)

        else:

            def dec(x):
                return "__%s__" % x

            parameters = list(condframe.columns) + list(kwargs.keys())
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
                                    dict(combs, **combinations), condframe, reference,
                                    **kwargs).drop(unwanted, axis=1)

    # ======================================================================================
    def pmf(self,
            potential,
            property,
            bins=10,
            interval=None,
            **kwargs):

        mics.verbose and info("PMF requested - %s case:" % self.method, self.title)
        mics.verbose and info("Reduced potential:", potential)
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
            mics.verbose and info("Bin[%d]:" % (i + 1), "%s = %s" % (property, str(zc)))
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
        for i in range(self.m):
            frame["state %s" % (i+1)] = np.histogram(y[i], bins, (ymin, ymax)).item(0)
        return frame
