import numpy as np
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify


def red(s):
    return "\033[1;31m" + s + "\033[0m"


def genfunc(expr, names, **kwargs):
    """
    Returns a function based on the passed argument.
    """
    if callable(expr):
        def func(x):
            return expr(x, **kwargs)
        return func

    elif isinstance(expr, str):
        try:
            variables = {}
            local_dict = kwargs.copy()
            for name in names:
                local_dict[name] = variables[name] = Symbol("x." + name)
            func = parse_expr(expr, local_dict)
        except SyntaxError:
            raise SyntaxError(red("unable to parse expression '%s'" % expr))
        if [s for s in func.free_symbols if s not in variables.values()]:
            raise ValueError(red("unspecified parameters found in expression '%s'" % expr))
        return lambdify("x", func, ["numpy"])

    else:
        raise ValueError(red("passed argument is neither a callable object nor a string"))


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
