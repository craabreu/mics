__version__ = "0.2.0"

from mics.MBAR import MBAR
from mics.MICS import MICS
from mics.mixtures import mixture
from mics.samples import pooledsample
from mics.samples import sample

pool = pooledsample  # TODO: remove this line and the 'pool' item below

verbose = False

__all__ = ['sample', 'pool', 'pooledsample', 'mixture', 'MICS', 'MBAR']
