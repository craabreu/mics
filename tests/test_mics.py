import numpy as np
import pandas as pd

import mics


def test_main():
    assert mics  # use your library here


m = 4
beta = 1.6773985789
data = ["tests/data/log_%d.dat" % (i + 1) for i in range(m)]

samples = mics.pool(verbose=True)
for i in range(m):
    dataset = pd.read_csv(data[i], sep=" ")
    potential = "beta*E%d" % (i + 1)
    difference = "beta*(E%d - E%d)" % (min(i+2, m), max(i, 1))
    samples.add(dataset, potential, difference, beta=beta)

neff = [100.829779921697, 76.82824014457174, 69.63811023389404, 55.179192164637165]
for i in range(4):
    np.testing.assert_almost_equal(samples[i].neff, neff[i])

# MICS

mixture = mics.MICS(samples, verbose=True)

fe = mixture.free_energies()
print(fe)

np.testing.assert_almost_equal(fe["f"][m], 3.6251084520815593)
np.testing.assert_almost_equal(fe["df"][m], 0.16158119695537948)

parameters = pd.DataFrame({"T": np.linspace(0.8, 1.2, 5)/(1.987E-3*beta)})

props = mixture.reweighting(potential="E4/(kB*T)",
                            properties={"E": "E4", "E2": "E4**2"},
                            derivatives={"Cv1": ("E", "T"), "dfdT": ("f", "T")},
                            combinations={"Cv2": "(E2 - E**2)/(kB*T**2)"},
                            conditions=parameters,
                            kB=1.987E-3)

print(props)

parameters = pd.DataFrame({"beta": beta*np.linspace(0.8, 1.2, 5)})


fu = mixture.reweighting(potential="beta*E4", conditions=parameters)
print(fu)

# MBAR

mbar = mics.MBAR(samples.copy().subsample(), verbose=True)
fe = mbar.free_energies()
print(fe)

props = mbar.reweighting(potential="beta*E4",
                         properties={"E": "PotEng + KinEng", "E2": "(PotEng + KinEng)**2"},
                         combinations={"Cv": "kB*beta**2*(E2 - E**2)"},
                         conditions=parameters,
                         kB=1.987E-3)
print(props)

fu = mbar.reweighting(potential="beta*E4", conditions=parameters)
print(fu)
