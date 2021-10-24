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
from myorbit import coord as co
from myorbit.util.general import angular_distance

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

def check_df(df, exp_df, exp_diff) :
    print (df[df.columns[0:8]])
    assert len(df) == len(exp_df)
    diff_secs = calc_diff_seconds(df, exp_df)
    assert diff_secs < exp_diff

TEST_DATA_PATH = Path(__file__).resolve().parents[0].joinpath('data')
    

if __name__ == "__main__" :
    None