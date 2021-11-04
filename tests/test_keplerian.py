"""
This module contains the tests for timeconv function
"""
# Standard library imports


# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np
from pathlib import Path



# Local application imports
import myorbit.planets as pl
from myorbit import coord as co
from myorbit.util.timeut import EQX_B1950, EQX_J2000
import myorbit.data_catalog as dc
from myorbit.orbits.ephemeris_input import EphemrisInput
import myorbit.util.timeut as ut
from myorbit.orbits.keplerian import KeplerianStateSolver

# The configuration file is shared between general config and logging config
CONFIG_INI=Path(__file__).resolve().parents[1].joinpath('conf','config.ini')
print (CONFIG_INI)
# For logging configuration
import logging.config
logging.config.fileConfig(CONFIG_INI, disable_existing_loggers=False)

from numpy.testing import assert_array_equal
from math import isclose

ABS=1e-12

def test_elliptical():
    T0_MJD = 56197.0
    solver = KeplerianStateSolver.make(tp_mjd=56198.22249000007, e=0.99999074, q=1.29609218)  
    r_xyz, rdot_xyz, r, h = solver.calc_rv(T0_MJD)
    assert r_xyz == approx(np.array([ 1.295960559873, -0.026122099794,0.] ), abs=ABS )
    assert rdot_xyz == approx(np.array([0.000215316642, 0.02136650003, 0.] ), abs=ABS ) 
    assert r == approx (1.2962237989036969,abs=ABS)
    assert h == approx (0.02769576586418702,abs=ABS)

def test_hiperbolical():
    solver = KeplerianStateSolver.make(tp_mjd=59311.54326000018, e=1.06388423, q=3.20746664)    
    T0_MJD = 56197.0
    r_xyz, rdot_xyz, r, h = solver.calc_rv(T0_MJD)
    assert r_xyz == approx(np.array([-14.33841087853 , -16.519465616568, 0.]), abs=ABS )
    assert rdot_xyz == approx(np.array([0.005049176396, 0.002730451155, 0.]) , abs=ABS ) 
    assert r == approx (21.87424903347594,abs=ABS)
    assert h == approx (0.04425936532200518,abs=ABS)


def test_parabolical():
    solver = KeplerianStateSolver.make(tp_mjd=57980.231000000145, e=1.0, q=2.48315593)    
    T0_MJD = 56197.0
    r_xyz, rdot_xyz, r, h = solver.calc_rv(T0_MJD)
    assert r_xyz == approx(np.array([ -9.14778995466 , -10.748293305451,0.]), abs=ABS )
    assert rdot_xyz == approx(np.array([0.005878285982, 0.002716096459, 0.]) , abs=ABS ) 
    assert r == approx (14.114101814660067,abs=ABS)
    assert h == approx (0.0383352619607468,abs=ABS)

