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
from myorbit.util.timeut import EQX_B1950, EQX_J2000
import myorbit.data_catalog as dc
from myorbit.ephemeris_input import EphemrisInput
import myorbit.util.timeut as ut
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


# The predictions for this one are very bad (1 minute precision)
def test_C_2007_M5_SOHO():  
    fn = TEST_DATA_PATH.joinpath('jpl_C_2007_M5_SOHO.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 492319
    EXP_DIFF_UNI = 492301
    EXP_DIFF_COWELLS = 257628.2
    EXP_DIFF_ENKES = 243121.9

    obj=dc.C_2007_M5_SOHO
    eph  = EphemrisInput(from_date="2006.04.01.0",
                        to_date = "2008.09.01.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(obj, eph)   
    check_df(df, exp_df, EXP_DIFF) 
    
    dfu = calc_eph_twobody_universal(obj, eph)
    check_df(dfu, exp_df, EXP_DIFF_UNI) 

    dfc = calc_eph_by_cowells(obj, eph)   
    check_df(dfc, exp_df, EXP_DIFF_COWELLS)     

    #dfc = calc_eph_by_enckes(obj, eph)   
    #check_df(dfc, exp_df, EXP_DIFF_ENKES)     


def test_C_2003_M3_SOHO():  
    fn = TEST_DATA_PATH.joinpath('jpl_C_2003_M3_SOHO.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 74286.96
    EXP_DIFF_UNI = 74285.66
    EXP_DIFF_COWELLS = 38009.9
    

    obj= dc.C_2003_M3_SOHO
    eph  = EphemrisInput(from_date="2001.03.01.0",
                        to_date = "2005.08.31.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(obj, eph)   
    check_df(df, exp_df, EXP_DIFF) 
    
    dfu = calc_eph_twobody_universal(obj, eph)
    check_df(dfu, exp_df, EXP_DIFF_UNI) 

    dfc = calc_eph_by_cowells(obj, eph)   
    check_df(dfc, exp_df, EXP_DIFF_COWELLS) 
