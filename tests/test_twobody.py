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
from myorbit.two_body import calc_eph_twobody, calc_eph_minor_body_perturbed
from myorbit.pert_cowels import calc_eph_by_cowells
from myorbit.pert_enckes import calc_eph_by_enckes

# The configuration file is shared between general config and logging config
CONFIG_INI=Path(__file__).resolve().parents[1].joinpath('conf','config.ini')
print (CONFIG_INI)
# For logging configuration
import logging.config
logging.config.fileConfig(CONFIG_INI, disable_existing_loggers=False)


from common import check_df, TEST_DATA_PATH

# Because ENCKES calcultations takes a lot of time, this flag variable is 
# to control when to run them
TEST_ENCKES = False

def test_HalleyB1950_for_1985():  
    fn = TEST_DATA_PATH.joinpath('jpl_halley_1985-Nov-15_1985-Apr-05.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 1305.1
    EXP_DIFF_PERT = 757
    EXP_DIFF_PERT_ENCKES = 771


    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.HALLEY_B1950, eph)   
    check_df(df, exp_df, EXP_DIFF) 

    df = calc_eph_minor_body_perturbed(dc.HALLEY_B1950, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT) 

    df = calc_eph_by_cowells(dc.HALLEY_B1950, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(dc.HALLEY_B1950, eph)   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    



def test_HalleyJ2000_for_1985():    
    """ For Halley, the epoch of the orbital elements for HALLEY_J2000 is 1994/02/17 i.e. the data were
    updated in that date (e.g. time for perihelion, Mean anomaly at epoch). In the case time for perihelion 
    has a difference of 4 days with the HALLEY_B1950 so using the non perturbing method in this case 
    are worse than the case of using Halley B1950. Howerver, using the perturbed method provides 
    good results because the epoch time is used as a starting, i.e., we start with very updated data and from that 
    point the integration procedure backward or forward is done. If the ephemeris date is close to the epoch of hte 
    orbital elements, the perturbed method will provide good results. Otherwise, the results will be less precise.
    """        
    fn = TEST_DATA_PATH.joinpath('jpl_halley_1985-Nov-15_1985-Apr-05.csv')
    
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 256239.92
    EXP_DIFF_PERT = 1570
    EXP_DIFF_PERT_ENCKES = 1570


    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.HALLEY_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF) 

    df = calc_eph_minor_body_perturbed(dc.HALLEY_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT) 

    df = calc_eph_by_cowells(dc.HALLEY_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT) 

    # TODO the lenght of the df result are not equal
    #df = calc_eph_by_enckes(dc.HALLEY_J2000, eph, 'comet')   
    #check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    
   



def test_HalleyJ2000_for_1997():    
    """ For Halley, the epoch of the orbital elements for HALLEY_J2000 is 1994/02/17 i.e. the data were
    updated in that date (e.g. time for perihelion, Mean anomaly at epoch). In the case time for perihelion 
    has a difference of 4 days with the HALLEY_B1950 so using the non perturbing method in this case 
    are worse than the case of using Halley B1950. However, using the perturbed method provides 
    good results because the epoch time is used as a starting, i.e., we start with very updated data and from that 
    point the integration procedure backward or forward is done. If the ephemeris date is close to the epoch of hte 
    orbital elements, the perturbed method will provide good results. Otherwise, the results will be less precise.
    """        
    fn = TEST_DATA_PATH.joinpath('jpl_halley_1997-Nov-15_1998-Apr-04.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 1099.5
    EXP_DIFF_PERT = 14
    EXP_DIFF_PERT_ENCKES = 8

    eph  = EphemrisInput(from_date="1997.11.15.0",
                        to_date = "1998.04.04.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.HALLEY_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF) 

    df = calc_eph_minor_body_perturbed(dc.HALLEY_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)     

    df = calc_eph_by_cowells(dc.HALLEY_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(dc.HALLEY_J2000, eph)   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    


def test_HalleyJ2000_for_2017():    
    """ For Halley, the epoch of the orbital elements for HALLEY_J2000 is 1994/02/17. In this test,
    the ephemeris is around  20 years in the future. As we can see, the differences increase with respect
    the ephemris calculated in test_HalleyJ2000_for_1997. Even though, the perturbed method works better that
    the non perturbed method.
    """        
    
    fn = TEST_DATA_PATH.joinpath('jpl_halley_2017-Nov-15_2018-Apr-04.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 1807    
    EXP_DIFF_PERT = 113.6
    EXP_DIFF_PERT_ENCKES = 66

    eph  = EphemrisInput(from_date="2017.11.15.0",
                        to_date = "2018.04.04.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.HALLEY_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF) 

    df = calc_eph_minor_body_perturbed(dc.HALLEY_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)     

    df = calc_eph_by_cowells(dc.HALLEY_J2000, eph, 'comet')   
    check_df(df, exp_df, EXP_DIFF_PERT)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(dc.HALLEY_J2000, eph, 'comet')   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    


def test_ceres_B1950_for_1992():
    """ Two tests are done here to show that in the case of CERES the perturbantions
     are very important. The epoch for CERES_B1950 is 1983.09.23.0 and the ephemeris
     are for 1992, i.e., around 10 years in the future. In the case of CERES_J2000
     the epoch is 2020.5.31.0 so it is not good idea to use CERES_J2000 for predict at 1992.
     This test is also done for doing the comparison.
    """    
    
    fn = TEST_DATA_PATH.joinpath('jpl_ceres_1992-Jun-27_1992-Jul-25.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 59888.3
    EXP_DIFF_PERTURBED = 300.6
    EXP_DIFF_PERTURBED_J2000 = 3322.6
    EXP_DIFF_PERT_ENCKES = 64

    eph = EphemrisInput(from_date="1992.06.27.0",
                        to_date = "1992.07.25.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.CERES_B1950, eph)   
    check_df(df, exp_df, EXP_DIFF)    

    df = calc_eph_minor_body_perturbed(dc.CERES_B1950, eph)
    check_df(df, exp_df, EXP_DIFF_PERTURBED)    

    df = calc_eph_minor_body_perturbed(dc.CERES_J2000, eph)
    check_df(df, exp_df, EXP_DIFF_PERTURBED_J2000)    

    df = calc_eph_by_cowells(dc.CERES_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF_PERTURBED_J2000)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(dc.CERES_J2000, eph)   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    


def test_ceres_J2000_for_2010():
    """ In this case, ceres_J2000 has an epoch of 2020.5.31.0. The predictions are 10 years backwards.
    The perturbed method works better that the nonperturbed method.
    """    
    fn = TEST_DATA_PATH.joinpath('jpl_ceres_2010-06-27_2010-07-25.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 30586
    EXP_DIFF_PERT = 841
    EXP_DIFF_PERT_ENCKES = 70
    
    eph = EphemrisInput(from_date="2010.06.27.0",
                        to_date = "2010.07.25.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.CERES_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF)        

    df = calc_eph_minor_body_perturbed(dc.CERES_J2000, eph)    
    check_df(df, exp_df, EXP_DIFF_PERT)        

    df = calc_eph_by_cowells(dc.CERES_J2000, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(dc.CERES_J2000, eph)   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    


def test_elliptical_C2012CH17_J2000_for_2012():
    # This comet  has an eccentricity of 0.999991
    fn = TEST_DATA_PATH.joinpath('jpl_C2012CH17_2012-Sep-27_2012-Nov-27.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 184
    EXP_DIFF_PERT = 106
    EXP_DIFF_PERT_ENCKES = 106

    eph = EphemrisInput(from_date="2012.09.27.0",
                        to_date = "2012.11.27.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")

    df = calc_eph_twobody(dc.C2012_CH17, eph)
    check_df(df, exp_df, EXP_DIFF)        

    df = calc_eph_minor_body_perturbed(dc.C2012_CH17, eph)    
    check_df(df, exp_df, EXP_DIFF_PERT)        

    df = calc_eph_by_cowells(dc.C2012_CH17, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(dc.C2012_CH17, eph)   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    
    
"""
def test_comet_with_twobodys_J2000():    

    
    fn = test_data_path.joinpath('jpl_halley_1985-Nov-15_1985-Apr-05.csv')
    exp_df = dc.read_jpl_data(fn)    

    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")

    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)

    df = calc_eph_twobody(HALLEY_J2000, eph, type='comet')
    assert len(df) == len(exp_df)
    assert calc_diff_seconds(df, exp_df) < 69418.96

    
def test_body_with_twobodys():    

    fn = test_data_path.joinpath('jpl_ceres_2020-May-15_2020-Jun-14.csv')
    exp_df = dc.read_jpl_data(fn)    

    eph = EphemrisInput(from_date="2020.05.15.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    CERES = dc.read_body_elms_for("Ceres",dc.DF_BODYS)

    df = calc_eph_twobody(CERES, eph, type='body')
    assert calc_diff_seconds(df, exp_df) < 2.4

    

# Elliptical comet
C2012_CH17 = read_comet_elms_for("C/2012 CH17 (MOSS)", DF_COMETS)   

# Hyperbolic comet
C_2020_J1_SONEAR = read_comet_elms_for("C/2020 J1 (SONEAR)", DF_COMETS) 

# Parabolic comet:
C_2018_F3_Johnson = read_comet_elms_for("C/2018 F3 (Johnson)", DF_COMETS) 



"""

def test_parabollic_C_2018_F3_Johnson_J2000_for_2017():
    # This comet follows a parabolic orbit, e=1
    fn = TEST_DATA_PATH.joinpath('jpl-C2018_F3_Johnson-Ago-01_2017-Ago-30.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 222
    EXP_DIFF_PERT = 20
    EXP_DIFF_PERT_ENCKES = 20

    eph = EphemrisInput(from_date="2017.8.01.0",
                        to_date = "2017.8.30.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")

    df = calc_eph_twobody(dc.C_2018_F3_Johnson, eph)
    check_df(df, exp_df, EXP_DIFF)        

    df = calc_eph_minor_body_perturbed(dc.C_2018_F3_Johnson, eph)    
    check_df(df, exp_df, EXP_DIFF_PERT)        

    df = calc_eph_by_cowells(dc.C_2018_F3_Johnson, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(dc.C_2018_F3_Johnson, eph)   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    


def test_hyperbolical_C_2020_J1_SONEAR_J2000_for_2020():
    # This comet follows a parabolic orbit, e=1
    fn = TEST_DATA_PATH.joinpath('jpl-C_2020_J1_SONEAR-Apr-01_2021-May-30.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 82
    EXP_DIFF_PERT = 81
    EXP_DIFF_PERT_ENCKES = 81
    obj = dc.C_2020_J1_SONEAR

    eph = EphemrisInput(from_date="2021.04.01.0",
                        to_date = "2021.05.30.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")

    df = calc_eph_twobody(obj, eph)
    check_df(df, exp_df, EXP_DIFF)        

    df = calc_eph_minor_body_perturbed(obj, eph)    
    check_df(df, exp_df, EXP_DIFF_PERT)        

    df = calc_eph_by_cowells(obj, eph)   
    check_df(df, exp_df, EXP_DIFF_PERT)    

    if TEST_ENCKES :
        df = calc_eph_by_enckes(obj, eph)   
        check_df(df, exp_df, EXP_DIFF_PERT_ENCKES)    

