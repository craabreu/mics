__version__ = "0.2.0"

from mics.MBAR import MBAR
from mics.MICS import MICS
from mics.mixtures import mixture
from mics.samples import pooledSample
from mics.samples import sample

pool = pooledSample  # TODO: remove this line and the 'pool' item below

__all__ = ['sample', 'pool', 'pooledSample', 'mixture', 'MICS', 'MBAR']
