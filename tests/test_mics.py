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

neff = [100.829779921697, 76.82824014457174, 69.63811023389404, 55.179192164637165]
for i in range(4):
    np.testing.assert_almost_equal(samples[i].neff, neff[i])

mixture = mics.MICS(samples, verbose=True)

fe = mixture.free_energies()
print(fe)

np.testing.assert_almost_equal(fe['f'][m-1], 3.6251084520815593)
np.testing.assert_almost_equal(fe['δf'][m-1], 0.16158119695537948)

parameters = pd.DataFrame({"beta": beta*np.linspace(0.8, 1.2, 5)})

props = mixture.reweighting(potential='beta*E4',
                            properties={'P': 'Press', 'E': 'PotEng + KinEng'},
                            conditions=parameters)

print(props)
