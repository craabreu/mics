from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import mics


def test_main():
    assert mics  # use your library here


mics.verbose = True
m = 4
beta = 1.6773985789
data = ["tests/data/log_%d.dat" % (i + 1) for i in range(m)]

samples = mics.pooledsample()
for i in range(m):
    dataset = pd.read_csv(data[i], sep=" ")
    potential = "beta*E%d" % (i + 1)
    autocorr = "beta*(E%d - E%d)" % (min(i+2, m), max(i, 1))
    samples += mics.sample(dataset, potential, autocorr, beta=beta)


def test_read_samples():
    for sample in samples:
        assert sample.dataset.columns[5] == 'Press'


def test_pooledsample():
    neff = [100.829779921697, 76.82824014457174, 69.63811023389404, 55.179192164637165]
    for i in range(m):
        assert samples[i].neff == pytest.approx(neff[i])


def test_samplesum():
    assert sum([sample for sample in samples]) == samples
    assert samples[0] + samples[1] == samples[0:2]
    assert samples[0] + samples[1:] == samples
    assert samples[0:2] + samples[2:4] == samples


def test_averaging():
    KE = [638.2521104699999, 637.60096689, 638.10787766, 638.76491318]
    dKE = [0.5450308320618747, 0.701666341463894, 0.5484439624133098, 0.5842767924919752]
    df = samples.averaging(dict(T='Temp', K='KinEng'), combinations=dict(R='K/T'))
    for i in range(m):
        assert df["K"][i] == pytest.approx(KE[i])
        assert df["dK"][i] == pytest.approx(dKE[i])


def test_mics_single_sample():
    dataset = pd.read_csv(data[0], sep=" ")
    sample = mics.sample(dataset, "beta*E1", "beta*(E2 - E1)", beta=beta)
    mixture = mics.mixture([sample], engine=mics.MICS())
    assert mixture.Overlap[0][0] == pytest.approx(1.0)


def test_mbar_single_sample():
    dataset = pd.read_csv(data[0], sep=" ")
    sample = mics.sample(dataset, "beta*E1", "beta*(E2 - E1)", beta=beta)
    mixture = mics.mixture([sample], mics.MBAR())
    assert mixture.Overlap[0][0] == pytest.approx(1.0)


mixture = samples.mixture(mics.MICS())
print(mixture.free_energies())


def test_mics_free_energies():
    fe = mixture.free_energies()
    assert fe["f"][m-1] == pytest.approx(3.6251084520815593)
    assert fe["df"][m-1] == pytest.approx(0.16158119695537948)


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

mbar = deepcopy(samples).subsampling().mixture(mics.MBAR())
print(mbar.free_energies())


def test_mbar_free_energies():
    fe = mbar.free_energies()
    assert fe["f"][m-1] == pytest.approx(3.657670266845165)
    assert fe["df"][m-1] == pytest.approx(0.1919766789868509)


props = mbar.reweighting(potential="beta*E4",
                         properties={"E": "PotEng + KinEng", "E2": "(PotEng + KinEng)**2"},
                         combinations={"Cv": "kB*beta**2*(E2 - E**2)"},
                         conditions=parameters,
                         kB=1.987E-3)
print(props)

fu = mbar.reweighting(potential="beta*E4", conditions=parameters)
print(fu)
