"""
.. module:: MICS
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`MICS`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
from numpy.linalg import multi_dot

from mics.utils import covariance
from mics.utils import cross_covariance
from mics.utils import info
from mics.utils import pinv
from mics.utils import safe_exp


class MICS:
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
    def __init__(mixture, composition=None):
        mixture.composition = composition

    # ======================================================================================
    def __initialize__(self, mixture, tol):
        m = mixture.m
        neff = mixture.neff
        verbose = mixture.verbose

        b = self.b = [s.b for s in mixture.samples]
        x = neff if self.composition is None else np.array(self.composition)
        pi = self.pi = x/np.sum(x)
        verbose and info("Mixture composition:", pi)

        verbose and info("Solving self-consistent equations...")
        iter = 1
        df = self._newton_raphson_iteration(mixture)
        if m > 1:
            verbose and info("Maximum deviation at iteration %d:" % iter, max(abs(df)))
            while any(abs(df) > tol):
                iter += 1
                mixture.f[1:m] += df
                df = self._newton_raphson_iteration(mixture)
                verbose and info("Maximum deviation at iteration %d:" % iter, max(abs(df)))
        verbose and info("Free energies after convergence:", mixture.f)

        self.Sp0 = sum(pi[i]**2*covariance(self.P[i], self.pm[i], b[i]) for i in range(m))
        mixture.Theta = multi_dot([self.iB0, self.Sp0, self.iB0])
        verbose and info("Free-energy covariance matrix:", mixture.Theta)

        mixture.Overlap = np.stack(self.pm)
        verbose and info("Overlap matrix:", mixture.Overlap)

    # ======================================================================================
    def _newton_raphson_iteration(self, mixture):
        m = mixture.m
        pi = self.pi
        n = mixture.n
        S = range(m)

        x = np.hstack(mixture.u)
        np.subtract((mixture.f + np.log(pi))[:, np.newaxis], x, out=x)
        xmax = np.amax(x, axis=0)
        np.subtract(x, xmax, out=x)
        np.exp(x, out=x)
        y = np.sum(x, axis=0)
        np.divide(x, y, out=x)
        np.log(y, out=y)
        np.add(xmax, y, out=y)
        np.negative(y, out=y)

        markers = np.cumsum(n[0:m-1])
        P = self.P = np.split(x, markers, axis=1)
        mixture.u0 = np.split(y, markers)
        self.pm = [np.mean(p, axis=1) for p in self.P]
        p0 = self.P0 = sum(pi[i]*self.pm[i] for i in S)
        self.B0 = np.diag(p0) - sum(pi[i]/n[i]*np.matmul(P[i], P[i].T) for i in S)
        self.iB0 = pinv(self.B0)
        df = np.matmul(self.iB0, pi - p0)
        return df[1:m] - df[0]

    # ======================================================================================
    def __reweight__(self, mixture, u, y, ref=0):
        S = range(mixture.m)
        pi = self.pi
        P = self.P
        pm = self.pm
        b = self.b

        w, argmax = safe_exp([mixture.u0[i] - u[i] for i in S])
        z = [w[i]*y[i] for i in S]

        iw0 = 1.0/sum(pi[i]*np.mean(w[i], axis=1) for i in S)[0]
        yu = sum(pi[i]*np.mean(z[i], axis=1) for i in S)*iw0
        fu = np.array([np.log(iw0) - argmax - mixture.f[ref]])

        r = [np.concatenate((z[i], w[i])) for i in S]
        rm = [np.mean(r[i], axis=1) for i in S]
        Sp0r0 = sum(pi[i]**2*cross_covariance(P[i], pm[i], r[i], rm[i], b[i]) for i in S)
        Sr0 = sum(pi[i]**2*covariance(r[i], rm[i], b[i]) for i in S)
        Ss0 = np.block([[self.Sp0, Sp0r0], [Sp0r0.T, Sr0]])

        pu = sum(pi[i]*np.mean(w[i]*P[i], axis=1) for i in S)*iw0
        pytu = sum(pi[i]*np.matmul(P[i], z[i].T)/mixture.n[i] for i in S)*iw0

        Dyup0 = np.matmul(self.iB0, np.outer(pu, yu) - pytu)
        Dyuz0 = np.diag(np.repeat(iw0, len(yu)))
        Dyuw0 = -yu[np.newaxis, :]*iw0

        pu[ref] -= 1.0
        Dfup0 = np.matmul(self.iB0, pu[:, np.newaxis])
        Dfuz0 = np.zeros([len(yu), 1])
        Dfuw0 = iw0

        G = np.block([[Dfup0, Dyup0],
                      [Dfuz0, Dyuz0],
                      [Dfuw0, Dyuw0]])

        Theta = multi_dot([G.T, Ss0, G])

        return np.concatenate([fu, yu]), Theta
