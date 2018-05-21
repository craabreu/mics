"""
.. module:: funcs
   :platform: Unix, Windows
   :synopsis: a module for dealing with functions

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

from mics.utils import InputError


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
