"""
.. module:: utils
   :platform: Unix, Windows
   :synopsis: a module for auxiliary tasks.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
from scipy.special import logsumexp

_msg_color = "\033[1;36m"
_val_color = "\033[0;36m"
_no_color = "\033[0m"


def multimap(functions, sample):
    """
    Applies a list of ``functions`` to DataFrame `sample` and returns a numpy matrix whose
    number of rows is equal to the length of list `functions` and whose number of columns
    is equal to the number of rows in `sample`.

    Note:
        Each function of the array might for instance receive `x` and return the result of
        an element-wise calculation involving `x['A']`, `x['B']`, etc, with 'A', 'B', etc
        being names of properties in DataFrame `sample`.

    """
    m = len(functions)
    n = sample.shape[0]
    f = np.empty([m, n])
    for i in range(m):
        f[i, :] = functions[i](sample).values
    return f


def covariance(y, ym, b):
    """
    Computes the covariance matrix of the rows of matrix `y` among themselves. The method
    of Overlap Batch Mean (OBM) is employed with blocks of size `b`.

    """
    S = _SumOfDeviationsPerBlock(y, ym, b)
    nmb = y.shape[1] - b
    return S.dot(S.T)/(b*nmb*(nmb + 1))


def cross_covariance(y, ym, z, zm, b):
    """
    Computes the cross-covariance matrix between the rows of matrix `y` with those of matrix
    `z`. The method of Overlap Batch Mean (OBM) is employed with blocks of size `b`.

    """
    Sy = _SumOfDeviationsPerBlock(y, ym, b)
    Sz = _SumOfDeviationsPerBlock(z, zm, b)
    nmb = y.shape[1] - b
    return Sy.dot(Sz.T)/(b*nmb*(nmb + 1))


def overlapSampling(u):
    """
    Computes the relative free energies of all sampled states using the Overlap Sampling
    method of Lee and Scott (1980).
    """
    m = len(u)
    seq = np.argsort([np.mean(u[i][i, :]) for i in range(m)])
    i = seq[0]
    f = np.zeros(m)
    for j in seq[1:m]:
        f[j] = f[i] + logsumexp(0.5*(u[j][j, :] - u[j][i, :])) - \
                      logsumexp(0.5*(u[i][i, :] - u[i][j, :]))
        i = j
    return f - f[0]


def _SumOfDeviationsPerBlock(y, ym, b):
    m, n = y.shape
    dy = y - ym.reshape([m, 1])
    B = np.empty([m, n-b+1])
    B[:, 0] = np.sum(dy[:, range(b)], axis=1)
    for j in range(n-b):
        B[:, j+1] = B[:, j] + dy[:, j+b] - dy[:, j]
    return B


def info(verbose, msg, val):
    if verbose:
        if isinstance(val, np.ndarray):
            print(_msg_color + msg + _val_color)
            x = val if val.ndim > 1 else val.reshape([len(val), 1])
            print(np.array2string(x) + _no_color)
        else:
            print(_msg_color + msg + _val_color, val, _no_color)
