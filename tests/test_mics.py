
import pandas as pd

import mics


def test_main():
    assert mics  # use your library here


beta = 1.6773985789

states = []
states.append(mics.states.State(pd.read_csv("tests/data/log_1.dat", sep=' '), lambda x: beta*x['E1'], lambda x: beta*x['E2']))
states.append(mics.states.State(pd.read_csv("tests/data/log_2.dat", sep=' '), lambda x: beta*x['E2']))
states.append(mics.states.State(pd.read_csv("tests/data/log_3.dat", sep=' '), lambda x: beta*x['E3']))
states.append(mics.states.State(pd.read_csv("tests/data/log_4.dat", sep=' '), lambda x: beta*x['E4']))

assert states[0].neff == 101
assert states[1].neff == 78
assert states[2].neff == 71
assert states[3].neff == 55

mix = mics.mixtures.Mixture(states, verbose=True)
