# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pandas as pd

import mics


def test_main():
    assert mics  # use your library here


m = 4
beta = 1.6773985789
data = ["tests/data/log_%d.dat" % (i + 1) for i in range(m)]


def solution_1():  # Closures

    def pot_gen(j):
        return lambda x: beta*x['E'+str(j+1)]

    def diff_gen(j):
        return lambda x: pot_gen(min(j+1, m-1))(x) - pot_gen(max(j-1, 0))(x)

    states = []
    for i in range(m):
        sample = pd.read_csv(data[i], sep=' ')
        potential = pot_gen(i)
        difference = diff_gen(i)
        states.append(mics.state(sample, potential, difference))

    return states


def solution_2():  # Partial evatualion:

    def pot_fun(x, j):
        return beta*x['E'+str(j+1)]

    def diff_fun(x, j):
        return pot_fun(x, min(j+1, m-1)) - pot_fun(x, max(j-1, 0))

    states = []
    for i in range(m):
        sample = pd.read_csv(data[i], sep=' ')
        potential = partial(pot_fun, j=i)
        difference = partial(diff_fun, j=i)
        states.append(mics.state(sample, potential, difference))

    return states


states = solution_2()

neff = [101, 76, 69, 54]
for i in range(4):
    assert states[i].neff == neff[i]

mixture = mics.mixture(states, verbose=True)

fe = mixture.free_energies()

np.testing.assert_almost_equal(fe['f'][m-1], 3.62444247539)
np.testing.assert_almost_equal(fe['δf'][m-1], 0.162805619863)
