# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import mics


def test_main():
    assert mics  # use your library here


m = 4
beta = 1.6773985789
data = ["tests/data/log_%d.dat" % (i + 1) for i in range(m)]


def potential(j):
    return lambda x: beta*x['E'+str(j+1)]


def difference(j):
    return lambda x: potential(min(j+1, m-1))(x) - potential(max(j-1, 0))(x)


states = []
for i in range(m):
    states.append(mics.state(pd.read_csv(data[i], sep=' '), potential(i), difference(i)))

neff = [101, 76, 69, 54]
for i in range(4):
    assert states[i].neff == neff[i]

mixture = mics.mixture(states, verbose=True)

fe = mixture.free_energies()

np.testing.assert_almost_equal(fe['f'][m-1], 3.62444247539)
np.testing.assert_almost_equal(fe['δf'][m-1], 0.162805619863)
