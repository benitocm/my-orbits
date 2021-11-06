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
from myorbit.util.timeut import EQX_B1950, EQX_J2000, mjd2str_date
import myorbit.data_catalog as dc
from myorbit.util.general import  my_range,  NoConvergenceError
from myorbit.kepler.keplerian import KeplerianStateSolver
from myorbit.ephemeris_input import EphemrisInput
from myorbit.two_body import calc_eph_twobody, calc_eph_minor_body_perturbed
from myorbit.pert_cowels import calc_eph_by_cowells
from myorbit.pert_enckes import calc_eph_by_enckes
from myorbit.kepler.near_parabolic import calc_stumpff_as_series, calc_stumpff_exact

# The configuration file is shared between general config and logging config
CONFIG_INI=Path(__file__).resolve().parents[1].joinpath('conf','config.ini')
print (CONFIG_INI)
# For logging configuration
import logging.config
logging.config.fileConfig(CONFIG_INI, disable_existing_loggers=False)


from common import check_df, TEST_DATA_PATH


ABS=1e-12

# The test consist should fail if a NoConvergenceError convergence error is raised
def test_almost_parabolical():
    delta_days=50
    df = dc.DF_COMETS
    COMETS_NO_CONVERGED = ['C/1680 V1', 'C/1843 D1 (Great March comet)', 'C/1882 R1-A (Great September comet)', 'C/1882 R1-B (Great September comet)', 'C/1882 R1-C (Great September comet)', 'C/1882 R1-D (Great September comet)', 'C/1963 R1 (Pereyra)', 'C/1965 S1-A (Ikeya-Seki)', 'C/1965 S1-B (Ikeya-Seki)', 'C/1967 C1 (Seki)', 'C/1970 K1 (White-Ortiz-Bolelli)', 'C/2004 V13 (SWAN)', 'C/2011 W3 (Lovejoy)', 'C/2013 G5 (Catalina)', 'C/2020 U5 (PANSTARRS)']
    df = df[df.Name.isin(COMETS_NO_CONVERGED)]
    for idx, name in enumerate(df['Name']): 
        obj = dc.read_comet_elms_for(name,df)
        msg = f'Testing Object: {obj.name} with Tp:{mjd2str_date(obj.tp_mjd)}'
        print (msg)
        solver = KeplerianStateSolver.make(e=obj.e, a=obj.a, tp_mjd=obj.tp_mjd, q=obj.q, epoch=obj.epoch_mjd)
        hs = []
        for clock_mjd in my_range(obj.tp_mjd-delta_days, obj.tp_mjd+delta_days, 2):               
            r_xyz, rdot_xyz, r, h_xyz, *others = solver.calc_rv(clock_mjd)
            hs.append(h_xyz)
            print(mjd2str_date(clock_mjd))    
        if not all(np.allclose(h_xyz, hs[0], atol=1e-12) for h_xyz in hs):
            msg = f'The angular momentum is NOT constant in the orbit'
            print (msg)

TEST_ENCKES = False

def test_C_2011_W3_Lovejoy_for_2011():  
    fn = TEST_DATA_PATH.joinpath('jpl_C_2011_W3_Lovejoy_2011-Nov-16_2011-Dic-16.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 493.9
    EXP_DIFF_PERT = 279.3
    EXP_DIFF_PERT_ENCKES = 279.3

    obj = dc.C_2011_W3_Lovejoy

    eph  = EphemrisInput(from_date="2011.10.16.0",
                        to_date = "2012.01.16.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(obj,eph)   
    check_df(df, exp_df, EXP_DIFF) 

    df = calc_eph_minor_body_perturbed(obj, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT) 

    df = calc_eph_by_cowells(obj, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(dc.HALLEY_B1950, eph)   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    



def test_stumpff():
    E=6.2831851329912345    
    exp_c1, exp_c2, exp_c3 = calc_stumpff_exact(E*E)
    c1,c2,c3 = calc_stumpff_as_series(E*E, epsilon=1e-10)
    EPSILON=1e-07
    assert c1 == approx(exp_c1, abs=EPSILON)
    assert c2 == approx(exp_c2, abs=EPSILON)
    assert c3 == approx(exp_c3, abs=EPSILON)
    print (c1, exp_c1)
    print (c2, exp_c2)
    print (c3, exp_c3)
    assert (c1>0) == (exp_c1>0)
    # Here the series one returns a negative number when for the input angle 
    # it should not, c2 cannot be negative for an angle near TWOPPI or 0
    assert (c2<0) == (exp_c2>0)
    assert (c3>0) == (exp_c3>0)


