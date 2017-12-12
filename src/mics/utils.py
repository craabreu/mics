"""
.. module:: utils
   :platform: Unix, Windows
   :synopsis: a module for auxiliary tasks.

.. moduleauthor:: Charlles R. A. Abreu <abreu@eq.ufrj.br>


"""

import numpy as np
import pandas as pd
from sympy import Matrix
from sympy import Symbol
from sympy import diff
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_tokenize import TokenError
from sympy.utilities.lambdify import lambdify


# ==========================================================================================
class InputError(Exception):
    def __init__(self, msg):
        super(InputError, self).__init__("\033[1;31m" + msg + "\033[0m")


# ==========================================================================================
def parse_func(function, symbols, constants):
    local_dict = symbols.copy()
    local_dict.update(constants)
    try:
        func = parse_expr(function, local_dict)
    except (SyntaxError, TokenError):
        raise SyntaxError("unable to parse function \"%s\"" % function)
    except Exception:
        raise InputError("unknown constants in function \"%s\"" % function)
    if type(func) is not float:
        if [s for s in func.free_symbols if s not in symbols.values()]:
            raise InputError("unknown symbols in function \"%s\"" % function)
    return func


# ==========================================================================================
def genfunc(function, variables, constants):
    """
    Returns a function based on the passed argument.
    """
    if callable(function):
        def func(x):
            return function(x, **constants)
        return func

    elif isinstance(function, str):
        symbols = dict((v, Symbol("x.%s" % v)) for v in variables)
        f = parse_func(function, symbols, constants)
        if f.free_symbols:
            return lambdify("x", f, ["numpy"])
        else:
            def func(x):
                return pd.Series(np.full(x.shape[0], f.evalf()))
            return func

    else:
        raise InputError("passed arg is neither a callable object nor a string")


# ==========================================================================================
def jacobian(functions, variables, constants):
    symbols = dict((v, Symbol("x[%d]" % i)) for (i, v) in enumerate(variables))
    f = Matrix([parse_func(expr, symbols, constants) for expr in functions])
    x = Matrix(list(symbols.values()))
    return lambdify("x", f), lambdify("x", f.jacobian(x))


# ==========================================================================================
def derivative(function, variable, symbols):
    local_dict = dict((x, Symbol(x)) for x in symbols)
    f = parse_expr(function, local_dict)
    return str(diff(f, Symbol(variable)))


# ==========================================================================================
def cases(conditions, constants, verbose):
    if conditions.empty:
        yield constants
    else:
        for (j, row) in conditions.iterrows():
            condition = row.to_dict()
            verbose and info("Condition[%d]:" % j, condition)
            condition.update(constants)
            yield condition


# ==========================================================================================
def multimap(functions, sample):
    """
    Applies a list of ``functions`` to DataFrame `sample` and returns a numpy matrix whose
    number of rows is equal to the length of list `functions` and whose number of columns
    is equal to the number of rows in `sample`.

    Note:
        Each function of the array might for instance receive `x` and return the result of
        an element-wise calculation involving `x["A"]`, `x["B"]`, etc, with "A", "B", etc
        being names of properties in DataFrame `sample`.

    """
    m = len(functions)
    n = sample.shape[0]
    f = np.empty([m, n])
    for i in range(m):
        f[i, :] = functions[i](sample).values
    return f


# ==========================================================================================
def covariance(y, ym, b):
    """
    Computes the covariance matrix of the rows of matrix `y` among themselves. The method
    of Overlap Batch Mean (OBM) is employed with blocks of size `b`.

    """
    S = _SumOfDeviationsPerBlock(y, ym, b)
    nmb = y.shape[1] - b
    return np.matmul(S, S.T)/(b*nmb*(nmb + 1))


# ==========================================================================================
def cross_covariance(y, ym, z, zm, b):
    """
    Computes the cross-covariance matrix between the rows of matrix `y` with those of matrix
    `z`. The method of Overlap Batch Mean (OBM) is employed with blocks of size `b`.

    """
    Sy = _SumOfDeviationsPerBlock(y, ym, b)
    Sz = _SumOfDeviationsPerBlock(z, zm, b)
    nmb = y.shape[1] - b
    return np.matmul(Sy, Sz.T)/(b*nmb*(nmb + 1))


# ==========================================================================================
def logsumexp(x):
    xmax = np.amax(x)
    return xmax + np.log(np.sum(np.exp(x - xmax), axis=0))


# ==========================================================================================
def overlapSampling(u):
    """
    Computes the relative free energies of all sampled states using the Overlap Sampling
    method of Lee and Scott (1980).
    """
    m = len(u)
    f = np.zeros(m)
    for j in range(1, m):
        i = j - 1
        f[j] = f[i] + logsumexp(0.5*(u[j][j, :] - u[j][i, :])) - \
            logsumexp(0.5*(u[i][i, :] - u[i][j, :]))
    return f - f[0]


# ==========================================================================================
def pinv(A):
    """
    Computes the Moore-Penrose pseudoinverse of a symmetric matrix using eigenvalue
    decomposition.

    """
    D, V = np.linalg.eigh(A)
    inv = np.vectorize(lambda x: 0.0 if np.isclose(x, 0.0) else 1.0/x)
    return np.matmul(V*inv(D), V.T)


# ==========================================================================================
def _SumOfDeviationsPerBlock(y, ym, b):
    m, n = y.shape
    dy = y - ym[:, np.newaxis]
    z = np.cumsum(dy, axis=1)
    B = np.empty([m, n-b+1])
    B[:, 0] = z[:, 0]
    B[:, 1:n-b+1] = z[:, b:n] - z[:, 0:n-b]
    return B


# ==========================================================================================
def info(msg, val=""):
    _msg_color = "\033[1;33m"
    _val_color = "\033[0;33m"
    _no_color = "\033[0m"
    if isinstance(val, np.ndarray):
        print(_msg_color + msg + _val_color)
        x = val if val.ndim > 1 else val[:, np.newaxis]
        print(np.array2string(x) + _no_color)
    else:
        print(_msg_color + msg + _val_color, val, _no_color)
