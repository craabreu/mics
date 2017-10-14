"""
.. module:: utils
   :platform: Unix, Windows
   :synopsis: a module for defining the class :class:`State`.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np


def multimap(functions, sample):
    """
    Applies a list of `functions` to DataFrame `sample` and returns a numpy matrix whose
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
    return np.matmul(S, S.T)/(b*nmb*(nmb + 1))
#    return Symmetric(syrk('U', 'T', 1.0/(b*nmb*(nmb+1)), S))


def cross_covariance(y, ym, z, zm, b):
    """
    Computes the cross-covariance matrix between the rows of matrix `y` with those of matrix
    `z`. The method of Overlap Batch Mean (OBM) is employed with blocks of size `b`.

    """
    Sy = _SumOfDeviationsPerBlock(y, ym, b)
    Sz = _SumOfDeviationsPerBlock(z, zm, b)
    nmb = y.shape[1] - b
    return np.matmul(Sy, Sz.T)/(b*nmb*(nmb + 1))
#    return gemm('T', 'N', 1.0/(b*nmb*(nmb+1)), Sy, Sz)


def _SumOfDeviationsPerBlock(y, ym, b):
    m, n = y.shape
    dy = y - ym
    B = np.empty([m, n-b+1])
    B[:, 0] = np.sum(dy[:, range(b)], axis=1)
    for j in range(n-b):
        B[:, j+1] = B[:, j] + dy[:, j+b] - dy[:, j]
    return B
