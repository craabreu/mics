"""
.. module:: funcs
   :platform: Unix, Windows
   :synopsis: a module for dealing with functions applicable to data frames

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


class func:
    """
    A class for storage, analysis, and evaluation of functions meant to
    be applied to Pandas data frames.

    Parameters
    ----------
    expression : str
        A character string defining the function
    variables : list(str)
        A list of strings containing the variables (or a superset thereof)
        of the defined function.
    constants : dict(str:Number), optional, default = {}
        A dictionary of constant names (str) associated to their values (Number)

    """
    def __init__(self, expression, variables, constants={}):
        symbols = dict((v, Symbol("x.%s" % v)) for v in variables)
        self.function = parse_func(expression, symbols, constants)
        self.expression = expression
        self.constants = constants

    def lambdify(self):
        """
        Returns a callable object that can be applied to a Pandas data frame.

        """
        if self.function.free_symbols:
            return lambdify("x", self.function, ["numpy"])
        else:
            value = self.function.evalf()

            def f(x):
                return pd.Series(np.full(x.shape[0], value))
            return f


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
