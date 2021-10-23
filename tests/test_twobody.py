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
from myorbit.util.general import angular_distance
from myorbit.two_body import calc_eph_twobody, calc_eph_minor_body_perturbed


def calc_diff_seconds(my_df, exp_df):
    my_df['r_AU_2'] = my_df['r[AU]']
    my_df['ra_2'] = my_df['ra'].map(co.make_ra)
    my_df['de_2'] = my_df['dec'].str.replace("'","m").str.replace('"',"s").map(co.make_lon)
    cols=['date','ra_2','de_2','r_AU_2']
    df = my_df[cols].copy()
    df = exp_df.merge(my_df, on='date')
    df['dist_ss'] = df.apply(lambda x: angular_distance(x['ra_1'],x['de_1'],x['ra_2'],x['de_2']), axis=1).map(np.rad2deg)*3600.0
    print (df['dist_ss'].abs() )
    print ((df['dist_ss'].abs()).sum())
    return (df['dist_ss'].abs()).sum()


""" The test data is obtained from https://ssd.jpl.nasa.gov/horizons/app.html#/
"""

TEST_DATA_PATH = Path(__file__).resolve().parents[0].joinpath('data')

def test_HalleyB1950_for_1985():    
    """[summary]
    """        
    fn = TEST_DATA_PATH.joinpath('jpl_halley_1985-Nov-15_1985-Apr-05.csv')

    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 1305.1
    EXP_DIFF_PERTURBED =761.92


    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.HALLEY_B1950, eph, obj_type='comet')   
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF

    df = calc_eph_minor_body_perturbed(dc.HALLEY_B1950, eph, type='comet')   
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF_PERTURBED

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
    EXP_DIFF_PERTURBED = 1570

    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.HALLEY_J2000, eph, obj_type='comet')   
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF

    df = calc_eph_minor_body_perturbed(dc.HALLEY_J2000, eph, type='comet')   
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF_PERTURBED

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
    EXP_DIFF_PERTURBED = 14

    eph  = EphemrisInput(from_date="1997.11.15.0",
                        to_date = "1998.04.04.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.HALLEY_J2000, eph, obj_type='comet')   
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF

    df = calc_eph_minor_body_perturbed(dc.HALLEY_J2000, eph, type='comet')   
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF_PERTURBED


def test_HalleyJ2000_for_2017():    
    """ For Halley, the epoch of the orbital elements for HALLEY_J2000 is 1994/02/17. In this test,
    the ephemeris is around  20 years in the future. As we can see, the differences increase with respect
    the ephemris calculated in test_HalleyJ2000_for_1997. Even though, the perturbed method works better that
    the non perturbed method.
    """        
    
    fn = TEST_DATA_PATH.joinpath('jpl_halley_2017-Nov-15_2018-Apr-04.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 1807    
    EXP_DIFF_PERTURBED = 113.6

    eph  = EphemrisInput(from_date="2017.11.15.0",
                        to_date = "2018.04.04.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.HALLEY_J2000, eph, obj_type='comet')   
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF

    df = calc_eph_minor_body_perturbed(dc.HALLEY_J2000, eph, type='comet')   
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF_PERTURBED


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

    eph = EphemrisInput(from_date="1992.06.27.0",
                        to_date = "1992.07.25.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.CERES_B1950, eph, obj_type='body')   
    print (df[df.columns[0:8]])

    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF

    df = calc_eph_minor_body_perturbed(dc.CERES_B1950, eph, type='body')
    print (df[df.columns[0:8]])

    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF_PERTURBED

    df = calc_eph_minor_body_perturbed(dc.CERES_J2000, eph, type='body')
    print (df[df.columns[0:8]])

    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF_PERTURBED_J2000


def test_ceres_J2000_for_2010():
    """ In this case, ceres_J2000 has an epoch of 2020.5.31.0. The predictions are 10 years backwards.
    The perturbed method works better that the nonperturbed method.
    """    
    fn = TEST_DATA_PATH.joinpath('jpl_ceres_2010-06-27_2010-07-25.csv')
    exp_df = dc.read_jpl_data(fn)    
    EXP_DIFF = 30586
    EXP_DIFF_PERTURBED = 841
    
    eph = EphemrisInput(from_date="2010.06.27.0",
                        to_date = "2010.07.25.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(dc.CERES_J2000, eph, obj_type='body')   
    print (df[df.columns[0:8]])

    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF

    df = calc_eph_minor_body_perturbed(dc.CERES_J2000, eph, type='body')
    print (df[df.columns[0:8]])

    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < EXP_DIFF_PERTURBED

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

    
    
"""