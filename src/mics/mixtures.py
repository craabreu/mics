"""
.. module:: mixtures
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`Mixture`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>

"""

import numpy as np
import pandas as pd

import mics
from mics.funcs import deltaMethod
from mics.funcs import diff
from mics.funcs import func
from mics.utils import InputError
from mics.utils import bennett
from mics.utils import cases
from mics.utils import crypto
from mics.utils import errorTitle
from mics.utils import info
from mics.utils import multimap
from mics.utils import propertyDict
from mics.utils import stdError


class mixture:
    """
    A mixture of independently collected samples (MICS).

    Parameters
    ----------
        samples : :class:`pooledsample` or list(:class:`sample`)
            A list of samples.
        engine : :class:`MICS` or :class:`MBAR`
            A method for mixture-model analysis.

    """

    def __init__(self, samples, engine):

        self.samples = samples
        self.engine = engine
        m = self.m = len(samples)
        if mics.verbose:
            # np.set_printoptions(precision=4, threshold=15, edgeitems=4, suppress=True)
            info("\n=== Setting up mixture ===")
            info("Analysis method: ", self.engine.__class__.__name__)
            info("Number of samples:", m)

        if m == 0:
            raise InputError("list of samples is empty")

        self.n = np.array([len(sample.dataset) for sample in samples])
        self.neff = np.array([sample.neff for sample in samples])
        names = self.names = list(samples[0].dataset.columns)
        if mics.verbose:
            info("Sample sizes:", self.n)
            info("Effective sample sizes:", self.neff)
            info("Properties:", ", ".join(names))

        potentials = [sample.potential.lambdify() for sample in samples]
        self.u = [multimap(potentials, sample.dataset) for sample in samples]
        self.f = bennett(self.u)
        mics.verbose and info("Initial free-energy guess:", self.f)
        self.engine.__initialize__(self)

    # ======================================================================================
    def __compute__(self, functions, constants):
        try:
            if isinstance(functions, str):
                funcs = [func(functions, self.names, constants).lambdify()]
            else:
                funcs = [func(f, self.names, constants).lambdify() for f in functions]
            return [multimap(funcs, sample.dataset) for sample in self.samples]
        except (InputError, KeyError):
            return None

    # ======================================================================================
    def free_energies(self, reference=0):
        """
        Computes the free energies of all sampled states relative to a given
        reference state, as well as their standard errors.

        Parameters
        ----------
            reference : int, optional, default=0
                Specifies which sampled state will be considered as a reference
                for computing free-energy differences.

        Returns
        -------
            pandas.DataFrame
                A data frame containing the free-energy differences and their
                computed standard errors for all sampled states.

        """
        frame = self.samples.__qualifiers__()
        frame["f"] = self.f - self.f[reference]
        T = self.Theta
        frame["df"] = np.sqrt(np.diag(T) - 2*T[:, reference] + T[reference, reference])
        return frame

    # ======================================================================================
    def reweighting(self, potential, properties={}, derivatives={}, combinations={},
                    conditions={}, reference=0, **constants):
        """
        Computes averages of specified properties at target states defined by
        a given reduced `potential` function with distinct passed parameter
        values, as well as the free energies of such states with respect to a
        sampled `reference` state. Also, computes derivatives of these averages
        and free energies with respect to the mentioned parameters. In addition,
        evaluates combinations of free energies, averages, and derivatives. In
        all cases, uncertainty propagation is handled automatically by means of
        the delta method.

        Parameters
        ----------
            potential : str
                A mathematical expression defining the reduced potential of the
                target states. It might depend on the collective variables of
                the mixture samples, as well as on external parameters whose
                values will be passed via `conditions` or `constants`, such as
                explained below.
            properties : dict(str: str), optional, default={}
                A dictionary associating names to mathematical expressions, thus
                defining a set of properties whose averages must be evaluated at
                the target states. If it is omitted, then only the relative free
                energies of the target states will be evaluated. The expressions
                might depend on the same collective variables and parameters
                mentioned above for `potential`.
            derivatives : dict(str: (str, str)), optional, default={}
                A dictionary associating names to (property, parameter) pairs,
                thus specifying derivatives of average properties at the target
                states or relative free energies of these states with respect
                to external parameters. For each pair, property must be either
                "f" (for free energy) or a name defined in `properties`, while
                parameter must be an external parameter such as described above
                for `potential`.
            combinations : dict(str: str), optional, default={}
                A dictionary associating names to mathematical expressions, thus
                defining combinations among average properties at the target
                states, the relative free energies of these states, and their
                derivatives with respect to external parameters. The expressions
                might depend on "f" (for free energy) or on the names defined in
                `properties`, as well as on external parameters such as described
                above for `potential`.
            conditions : pandas.DataFrame or dict, optional, default={}
                A data frame whose column names are external parameters present
                in mathematical expressions specified in arguments `potential`,
                `properties`, and `combinations`. The rows of the data frame
                contain sets of values of these parameters, in such as way that
                the reweighting is carried out for every single set. This is a
                way of defining multiple target states from a single `potential`
                expression. The same information can be passed as a dictionary
                associating names to lists of numerical values, provided that
                all lists are equally sized. If it is empty, then a unique
                target state will be considered and all external parameters in
                `potential`, if any, must be passed as keyword arguments.
            reference : int, optional, default=0
                The index of a sampled state to be considered as a reference for
                computing relative free energies.
            **constants : keyword arguments
                A set of keyword arguments passed as name=value, aimed to define
                external parameter values for the evaluation of mathematical
                expressions. These values will be repeated at all target states
                specified via `potential` and `conditions`.

        Returns
        -------
            pandas.DataFrame
                A data frame containing the computed quantities, along with
                their estimated uncertainties, at all target states specified
                via `potential` and `conditions`.

        """
        if mics.verbose:
            info("\n=== Performing reweighting with %s ===" % self.engine.__class__.__name__)
            info("Reduced potential:", potential)
            constants and info("Provided constants: ", constants)

        freeEnergy = "f"
        if freeEnergy in properties.keys():
            raise InputError("Word % is reserved for free energies" % freeEnergy)
        condframe = pd.DataFrame(data=conditions) if isinstance(conditions, dict) else conditions
        propfuncs = list(properties.values())

        if not derivatives:
            propnames = [freeEnergy] + list(properties.keys())
            combs = combinations.values()

            gProps = self.__compute__(propfuncs, constants)
            if combinations:
                gDelta = deltaMethod(combs, propnames, constants)

            results = list()
            for (index, condition) in cases(condframe):
                mics.verbose and condition and info("Condition[%s]" % index, condition)
                consts = dict(condition, **constants)
                u = self.__compute__(potential, consts)
                y = gProps if gProps else self.__compute__(propfuncs, consts)
                (yu, Theta) = self.engine.__reweight__(self, u, y, reference)
                result = propertyDict(propnames, yu, stdError(Theta))
                if combinations:
                    delta = gDelta if gDelta.valid else deltaMethod(combs, propnames, consts)
                    (h, dh) = delta.evaluate(yu, Theta)
                    result.update(propertyDict(combinations.keys(), h, dh))
                results.append(result.to_frame(index))

            return condframe.join(pd.concat(results))

        else:
            symbols = list(condframe.columns) + list(constants.keys())
            parameters = set(x for (y, x) in derivatives.values())
            props = dict()
            for x in parameters:
                props[crypto(x)] = diff(potential, x, symbols)
            combs = dict()
            for (z, (y, x)) in derivatives.items():
                if y == freeEnergy:
                    combs[z] = crypto(x)
                else:
                    dydx = diff(properties[y], x, symbols)
                    props[crypto(z)] = "%s - (%s)*(%s)" % (dydx, props[crypto(x)], properties[y])
                    combs[z] = "%s + (%s)*(%s)" % (crypto(z), crypto(x), y)
            unwanted = sum([[x, errorTitle(x)] for x in props.keys()], [])
            return self.reweighting(potential, dict(properties, **props), {},
                                    dict(combs, **combinations), condframe, reference,
                                    **constants).drop(unwanted, axis=1)

    # ======================================================================================
    def pmf(self, potential, property, bins=10, interval=None, **constants):

        if mics.verbose:
            info("\n=== Computing PMF with %s ===" % self.engine.__class__.__name__)
            info("Reduced potential:", potential)
        u = self.__compute__(potential, constants)
        z = self.__compute__(property, constants)
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
            (yu, Theta) = self.engine.__reweight__(self, u, y)
            if yu[1] > 0.0:
                dyu = np.sqrt(max(0.0, Theta[1, 1]))
                results.append([zc, -np.log(yu[1]), dyu/yu[1]])

        return pd.DataFrame(results, columns=[property, "pmf", errorTitle("pmf")])

    # ======================================================================================
    def histograms(self, property="u0", bins=100, **constants):
        if property == "u0":
            y = self.u0
        elif property == "state":
            w = np.arange(self.m) + 1
            wsum = sum(w)
            y = [wsum*np.average(p, axis=0, weights=w) for p in self.P]
        elif property == "potential":
            y = [self.u[i][i, :] for i in range(self.m)]
        else:
            y = self.__compute__(property, constants)
        ymin = min([np.amin(x) for x in y])
        ymax = max([np.amax(x) for x in y])
        delta = (ymax - ymin)/bins
        center = [ymin + delta*(i + 0.5) for i in range(bins)]
        frame = pd.DataFrame({property: center})
        for i in range(self.m):
            frame["state %s" % (i+1)] = np.histogram(y[i], bins, (ymin, ymax))[0]
        return frame
