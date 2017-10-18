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


def pot_fun(x, beta, j):
    return beta*x['E'+str(j+1)]


def diff_fun(x, beta, j):
    return pot_fun(x, beta, min(j+1, m-1)) - pot_fun(x, beta, max(j-1, 0))


samples = []
for i in range(m):
    dataset = pd.read_csv(data[i], sep=' ')
    potential = partial(pot_fun, beta=beta, j=i)
    difference = partial(diff_fun, beta=beta, j=i)
    samples.append(mics.sample(dataset, potential, difference))

neff = [100.53337462306746, 75.96158869910701, 68.72831124139921, 54.195291583870194]
for i in range(4):
    np.testing.assert_almost_equal(samples[i].neff, neff[i])

mixture = mics.mixture(samples, verbose=True)

fe = mixture.free_energies()
print(fe)

np.testing.assert_almost_equal(fe['f'][m-1], 3.6245656740094492)
np.testing.assert_almost_equal(fe['δf'][m-1], 0.16278496395668807)

properties = ['Press', 'PotEng']
potential = partial(pot_fun, j=m-1)

props = mixture.reweight(properties, potential, parameter=[0.9*beta, beta, 1.1*beta])
