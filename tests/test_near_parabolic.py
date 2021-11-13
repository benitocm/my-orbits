"""
This module contains the tests for timeconv function
"""
# Standard library imports


# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np
from pathlib import Path
import sys


# Local application imports
from myorbit.util.timeut import EQX_B1950, EQX_J2000, mjd2str_date
import myorbit.data_catalog as dc
from myorbit.util.general import  my_range,  NoConvergenceError
from myorbit.kepler.keplerian import KeplerianStateSolver
from myorbit.ephemeris_input import EphemrisInput
from myorbit.two_body import calc_eph_twobody, calc_eph_minor_body_perturbed, calc_eph_twobody_universal
from myorbit.pert_cowels import calc_eph_by_cowells
from myorbit.pert_enckes import calc_eph_by_enckes

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
        solver = KeplerianStateSolver.make(e=obj.e, a=obj.a, tp_mjd=obj.tp_mjd, q=obj.q, epoch=obj.epoch_mjd, force_orbit='near_parabolical')
        hs = []
        for clock_mjd in my_range(obj.tp_mjd-delta_days, obj.tp_mjd+delta_days, 2):               
            r_xyz, rdot_xyz, r, h_xyz, *others = solver.calc_rv(clock_mjd)
            hs.append(h_xyz)
            #print(mjd2str_date(clock_mjd))    
        if not all(np.allclose(h_xyz, hs[0], atol=1e-12) for h_xyz in hs):
            msg = f'The angular momentum is NOT constant in the orbit'
            print (msg)

TEST_ENCKES = True

def test_C_2011_W3_Lovejoy_for_2011():  
    fn = TEST_DATA_PATH.joinpath('jpl_C_2011_W3_Lovejoy_2011-Nov-16_2011-Dic-16.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 493.9
    EXP_DIFF_NEAR_PARABOLICAL = 493.9
    EXP_DIFF_PERT = 279.3
    EXP_DIFF_PERT_ENCKES = 290.99
    FUNC_NAME=sys._getframe().f_code.co_name

    obj = dc.C_2011_W3_Lovejoy

    eph  = EphemrisInput(from_date="2011.10.16.0",
                        to_date = "2012.01.16.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(obj,eph)   
    method=FUNC_NAME+":calc_eph_twobody"
    check_df(df, exp_df, EXP_DIFF, method) 

    df = calc_eph_twobody(obj,eph,force_orbit='near_parabolical')   
    method=FUNC_NAME+":calc_eph_twobody_near_parabolical"
    check_df(df, exp_df, EXP_DIFF_NEAR_PARABOLICAL,method) 

    df = calc_eph_minor_body_perturbed(obj, eph)   
    method=FUNC_NAME+":calc_eph_minor_body_perturbed"
    check_df(df, exp_df, EXP_DIFF_PERT,method) 

    df = calc_eph_by_cowells(obj, eph)   
    method=FUNC_NAME+":calc_eph_by_cowells"
    check_df(df, exp_df, EXP_DIFF_PERT,method)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(obj, eph)   
        method=FUNC_NAME+":calc_eph_by_enckes"
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES,method)    


def test_C_2007_M5_SOHO_at_perihelion():  
    fn = TEST_DATA_PATH.joinpath('jpl_C_2007_M5_SOHO_at_perihelion.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF_NEAR_PARABOLICAL = 1501.1
    EXP_DIFF_PARABOLICAL = 1501.1
    EXP_DIFF_UNIVERSAL = 1501.1
    EXP_DIFF_PERT = 464.3
    EXP_DIFF_PERT_ENCKES = 632.8
    FUNC_NAME=sys._getframe().f_code.co_name
    
    obj=dc.C_2007_M5_SOHO
    eph  = EphemrisInput(from_date="2007.06.15.0",
                        to_date = "2007.07.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)
      
    df = calc_eph_twobody(obj, eph)   
    method=FUNC_NAME+":calc_eph_twobody"
    check_df(df, exp_df, EXP_DIFF_PARABOLICAL,method) 
    
    df = calc_eph_twobody(obj, eph, force_orbit='near_parabolical')   
    method=FUNC_NAME+":calc_eph_twobody_near_parabolical"
    check_df(df, exp_df, EXP_DIFF_NEAR_PARABOLICAL,method) 
    
    df = calc_eph_twobody_universal(obj, eph)   
    method=FUNC_NAME+":calc_eph_twobody_universal"
    check_df(df, exp_df, EXP_DIFF_UNIVERSAL,method) 
    
    df = calc_eph_by_cowells(obj, eph)  
    method=FUNC_NAME+":calc_eph_by_cowells" 
    check_df(df, exp_df, EXP_DIFF_PERT,method)    
    
    if TEST_ENCKES :
        df = calc_eph_by_enckes(obj, eph)   
        method=FUNC_NAME+":calc_eph_by_enckes"
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES,method)    
    
    
    
def test_C_2007_M5_SOHO_6_months():  
    fn = TEST_DATA_PATH.joinpath('jpl_C_2007_M5_SOHO_6months.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF_NEAR_PARABOLICAL = 60849.4
    EXP_DIFF_PARABOLICAL = 60849.4
    EXP_DIFF_UNIVERSAL = 60849.3
    EXP_DIFF_PERT = 18467.9
    EXP_DIFF_PERT_ENCKES = 18968.1
    FUNC_NAME=sys._getframe().f_code.co_name
    
    obj=dc.C_2007_M5_SOHO
    eph  = EphemrisInput(from_date="2007.03.15.0",
                        to_date = "2007.10.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)
      
    df = calc_eph_twobody(obj, eph)   
    method=FUNC_NAME+":calc_eph_twobody"
    check_df(df, exp_df, EXP_DIFF_PARABOLICAL,method) 
    
    
    df = calc_eph_twobody(obj, eph, force_orbit='near_parabolical')   
    method=FUNC_NAME+":calc_eph_twobody_near_parabolical"
    check_df(df, exp_df, EXP_DIFF_NEAR_PARABOLICAL,method) 
    
    df = calc_eph_twobody_universal(obj, eph)   
    method=FUNC_NAME+":calc_eph_twobody_universal"
    check_df(df, exp_df, EXP_DIFF_UNIVERSAL,method) 
    
    df = calc_eph_by_cowells(obj, eph)   
    method=FUNC_NAME+":calc_eph_by_cowells" 
    check_df(df, exp_df, EXP_DIFF_PERT,method)    
    
    if TEST_ENCKES :
        df = calc_eph_by_enckes(obj, eph)   
        method=FUNC_NAME+":calc_eph_by_enckes"
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES,method)        
    
    
    

