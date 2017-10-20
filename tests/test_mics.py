# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import mics


def test_main():
    assert mics  # use your library here


m = 4
beta = 1.6773985789
data = ["tests/data/log_%d.dat" % (i + 1) for i in range(m)]


samples = []
for i in range(m):
    dataset = pd.read_csv(data[i], sep=' ')
    potential = "beta*E%d" % (i + 1)
    difference = "beta*(E%d - E%d)" % (min(i+2, m), max(i, 1))
    samples.append(mics.sample(dataset, potential, difference, beta=beta))

neff = [100.53337462306746, 75.96158869910701, 68.72831124139921, 54.195291583870194]
for i in range(4):
    np.testing.assert_almost_equal(samples[i].neff, neff[i])

mixture = mics.mixture(samples, verbose=True)

fe = mixture.free_energies()
print(fe)

np.testing.assert_almost_equal(fe['f'][m-1], 3.6245656740094492)
np.testing.assert_almost_equal(fe['δf'][m-1], 0.16278496395668807)

parameters = pd.DataFrame({"beta": beta*np.array([0.9, 1.0, 1.1])})


def potential(x, **p):
    return p['beta']*x['E4']


props = mixture.reweight(potential,
                         properties={'P': 'Press', 'U': 'PotEng'},
                         conditions=parameters)
