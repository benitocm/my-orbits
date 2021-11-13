"""
This module contains functions to access vsop87 data
"""

# Standard library imports
import logging
from functools import partial
from pathlib import Path
from configparser import ConfigParser

# Third party imports
import pandas as pd
import numpy as np
from toolz import pipe 
from numpy import sin, cos, deg2rad, rad2deg, tan

# Local application imports
from myorbit.util import timeut as tc
from myorbit.util.timeut import JD_J2000, JD_B1950, CENTURY
from myorbit import coord as co
from myorbit.util.general import  memoize, kahan_sum
from myorbit.coord import polarFcartesian, Coord, polarFcartesian, prec_mtx_equat
from myorbit.data_catalog import CometElms
from myorbit.util.constants import *

logger = logging.getLogger(__name__)

#np.set_printoptions(precision=15)
#VSOP87A - rectangular, of J2000.0
#VSOP87B - spherical, of J2000.0
#VSOP87C - rectangular, equinox of date
#VSOP87D - spherical, equinox of date
#VSOP87E - rectangular, barycentric, of J2000.0

CONFIG_INI=Path(__file__).resolve().parents[3].joinpath('conf','config.ini')
cfg = ConfigParser()
cfg.read(CONFIG_INI)
VSOP87_DATA_DIR=Path(cfg.get('general','vso87_data_dir_path'))

def reduce_rad(rad, to_positive=False):
    remainder = tc.my_frac(rad/TWOPI)*TWOPI
    if rad > 0 :
        return remainder
    else :
        return -remainder + TWOPI if to_positive else -remainder        

@memoize
def read_file(fn) :
    mtxs = {}
    sname = ''    
    rows = []
    with open(fn,'rt') as f:
        for i, line in enumerate(f):
            if i > 50000:
                break
            line = line.strip()
            if line.startswith('VSOP87'):
                if len(rows) > 0 :
                    df = np.array( rows)   
                    mtxs[sname] = df
                    rows = []
                if 'VARIABLE 1 (LBR)' in line :
                    sname = 'L'
                elif 'VARIABLE 2 (LBR)' in line :
                    sname = 'B'
                elif 'VARIABLE 3 (LBR)' in line :
                    sname = 'R'
                elif 'VARIABLE 1 (XYZ)' in line :
                    sname = 'X'
                elif 'VARIABLE 2 (XYZ)' in line :
                    sname = 'Y'
                elif 'VARIABLE 3 (XYZ)' in line :
                    sname = 'Z'
                else :
                    sname = 'UNKNOWN'
                sname += line.split()[7].replace('*','').replace('T','')
            else :
                row = np.array(line.split()[-3:],np.double)
                #print (row)
                rows.append(row)
        if len(rows) > 0 :
                df = np.array( rows)
                mtxs[sname] = df
    return mtxs
    
 

def vsop_pol(x=0, a0=0, a1=0, a2=0, a3=0, a4=0, a5=0) :
    """
    Up to x5
    """
    v =  a0 + x*(a1+x*(a2+x*(a3+x*(a4+a5*x))))
    return v
    
def p_rad(alpha):
    print (np.rad2deg(alpha))

def p_radv(v):
    print(f'r: {v[0]}  lon: {rad2deg(v[1])}  lat: {rad2deg(v[2])}')

def p_eq(v):
    c = Coord(v,'', co.EQUAT2_TYPE)    
    print (co.as_str(c))

def do_sum (tau, mtx_dict, var_name):
    mtx = mtx_dict[var_name] 
    return kahan_sum(mtx[:,0]*np.cos(mtx[:,1]+mtx[:,2]*tau))

def do_calc(var_prefix, mtx_dict, tau):
    # var_prefix='X'
    # needs to iterate from X0, X1, X2, X3, X4, X5
    func = partial (do_sum, tau, mtx_dict)
    keys = [k for k in mtx_dict.keys() if k.startswith(var_prefix)]
    # 'X0', 'X1', 'X2', 'X3'
    result_tup = tuple(map(func, keys))
    return vsop_pol(tau,*result_tup) 
 
def calc_series (fn, jde, var_names):
    tau = (jde- 2451545.0) / 365250.0
    mtx_dict = read_file(fn)
    var_result = [do_calc(var_prefix, mtx_dict, tau) for var_prefix in var_names]
    return var_result[0], var_result[1], var_result[2]


"""
VSOP87 provides methos to calculate Heliocentric coordinates:
    L, the ecliptical longitude (different from the orbtial longitude)
    B, the ecliptical latitude
    R, the radius vector (=distance to the Sun)
"""
        
def h_xyz_eclip_eqxdate(name, jde):
    sfx = name.lower()[:3]
    fn=VSOP87_DATA_DIR.joinpath('VSOP87C.'+sfx)
    x, y, z = calc_series(fn, jde,['X','Y','Z'])    
    return np.array([x,y,z])

def h_xyz_eclip_j2000(name, jde):
    sfx = name.lower()[:3]
    fn=VSOP87_DATA_DIR.joinpath('VSOP87A.'+sfx)
    x, y, z = calc_series(fn, jde,['X','Y','Z'])    
    return np.array([x,y,z])


def correction(T, lon_sun):
    return lon_sun - np.deg2rad(1.397)*T - np.deg2rad(0.00031)*T*T

def do_fk5(l, b, jde):
    T = (jde - JD_J2000) / 36525.0
    lda = l - deg2rad(1.397)*T - deg2rad(0.00031)*T*T
    delta_lon = -deg2rad(0.09033/3600) + deg2rad(0.03916/3600)*(cos(lda)+sin(lda))*tan(b)
    delta_lat = deg2rad(0.03916/3600)*(np.cos(lda)- np.sin(lda))
    l += delta_lon
    b += delta_lat
    return l,b

def h_rlb_eclip_eqxdate(name, jde, tofk5=False):
    """ 
    Computes the orbital elements of an elliptical orbit from position
    and velocity vectors
    
    Args:
        name : Name of the planet 
        jde  :  Julian day of the ephemeris
        tofk5 : To indicate if the result has to be transformed to FK5 coordinate system
        
    Returns :
        A numpy vector (3) with:
            R : radio vector
            L : Heliocentric Ecliptical longitude of the planet
            B : Heliocentric Ecliptical latitude of the planet
    """

    sfx = name.lower()[:3]
    fn=VSOP87_DATA_DIR.joinpath('VSOP87D.'+sfx)
    r, l, b = calc_series(fn, jde,['R','L','B'])    
    l = reduce_rad(l,True) # longitude muast be [0,360]
    b = reduce_rad(b,False) # latitud must be [-90,90]
    # Up to know, l and b are referred to the mean dynamical
    # ecliptic and equinox of the date defined by VSOP.
    # It differs very slightly from the standdard FK5 (pg. 219 Meedus)
    if tofk5 :
        l,b = do_fk5(l,b,jde)
    return np.array([r,l,b])


def h_rlb_eclip_j2000(name, jde):
    sfx = name.lower()[:3]
    fn=VSOP87_DATA_DIR.joinpath('VSOP87B.'+sfx)
    r, l, b = calc_series(fn, jde,['R','L','B'])    
    return np.array([r,reduce_rad(l,True),reduce_rad(b,False)])


def g_rlb_eclip_sun_eqxdate(jde, tofk5=True) : 
    """
    Computes the GEOMETRIC (not icnldue nutaiton an abberration)
    equatorial cartesian coordinates of the sun
    """
    h_rlb_earth_eqxdate = h_rlb_eclip_eqxdate("Earth",jde, False)
    #p_radv(h_rlb_earth_eqxdate)
    r = h_rlb_earth_eqxdate[0]
    l = h_rlb_earth_eqxdate[1] + PI
    b = -h_rlb_earth_eqxdate[2]    
    T = (jde - JD_J2000) / 36525
    if tofk5 :
        T = (jde - 2451545.0) / 36525
        lda = correction(T,l)    
        #p_rad(lda)
        delta_lon = -deg2rad(0.09033/3600)
        #p_rad(delta_lon)
        delta_lat = deg2rad(0.03916/3600)*(cos(lda)-sin(lda))
        #p_rad(delta_lat)
        l += delta_lon
        b += delta_lat
    return np.array([r,l,b])


def g_xyz_equat_sun_eqxdate(jde, tofk5=True) : 
    """
    Computes the GEOMETRIC (not include nutation not abberration)
    equatorial cartesian coordinates of the sun
    Reference to the mean equator and equinox of the date.
    Used for minor planets
    """
    g_rlb = g_rlb_eclip_sun_eqxdate(jde,tofk5)
    T = (jde - JD_J2000) / 36525
    # It is refered to the mean equinox of the dates so
    # the obliquit is calcualted in the same instant jde as centuries
    # pag 173 Meeus
    # When expdate is used, we can use the normal matrix to pass to equatorial
    # but when we are using j2000 versions we need to use the special matrix
    obl = co.obliquity (T)
    p_rad(obl)
    mtx = co.Rx_3d(-obl)
    return mtx.dot(co.cartesianFpolar(g_rlb))


def g_xyz_vsopeclip_sun_j2000(jde) : 
    """
    Returns the geocentric ecliptic cartesian coordinates of the sun
    in VSOP ecliptic system (ecliptic dynamical equinox J2000)
    pg 175 
    """

    h_rlb_earth_j2000= h_rlb_eclip_j2000("Earth",jde)
    r = h_rlb_earth_j2000[0]
    l = h_rlb_earth_j2000[1] + PI
    b = -h_rlb_earth_j2000[2]    
    return co.cartesianFpolar(np.array([r,l,b]))


## Matrix to transform from the VSOP frame of reference to equatorial FK5
MTX_FK5EQUAT_F_VSOPFR = np.array([
    [ 1.000000000000, 0.000000440360, -0.000000190919],
    [-0.000000479966, 0.917482137087, -0.397776982902],
    [ 0.000000000000, 0.397776982902,  0.917482137087]
])

def g_xyz_equat_sun_j2000 (jde) : 
    """
    Returns the geo equatorial FK5 frame J2000
    """
    h_rlb_earth_eqxdate= h_rlb_eclip_j2000("Earth",jde)
    r = h_rlb_earth_eqxdate[0]
    l = h_rlb_earth_eqxdate[1] + PI
    b = -h_rlb_earth_eqxdate[2]    
    xyz = co.cartesianFpolar(np.array([r,l,b]))
    # Up to now we are still in the eclipticla dynamical ereference frame (VSOP) 
    # of J2000.0. To transform FK5 J2000.0 reference we need to .dot  with
    # a matrix.
    return MTX_FK5EQUAT_F_VSOPFR.dot(xyz)

def g_xyz_equat_sun_at_other_mean_equinox (jde, T) : 
    """
    Returns the geo equatorial coordinates system of the sun at 
    any other mean equinox
    T is the century of that equinox
    The result are referred to the mean equinox of an epoch which differs 
    from date for which the values are calculated.
    """
    # we use the special matrix to transform J2000 Fk5
    xyz = g_xyz_equat_sun_j2000(jde)
    # once were J2000 fk5, we can apply the normal precesion matrix to precess
    T1 = 0 # we are in J2000
    T2 = T
    return prec_mtx_equat(T1,T2).dot(xyz)


def g_xyz_eclip_planet_eqxdate(name, jde):
    h_xyz_eclipt_earth = h_xyz_eclip_eqxdate("Earth",jde)
    h_xyz_eclipt_planet = h_xyz_eclip_eqxdate(name,jde)
    return h_xyz_eclipt_planet - h_xyz_eclipt_earth

def g_rlb_equat_planet_J2000(name, jde):
    """
    It seems that to pass from VSOP eclipt to equat instead of applytin
    the normal matrix, we need to use the an special matrix MTX_FK5EQUAT_F_VSOPPRF
    The results are similar but Meeus say that at pg 174
    """
    T = (jde - JD_J2000) / CENTURY
    return pipe(g_xyz_eclip_planet_eqxdate(name, jde),
                #co.mtx_equatFeclip(T).dot,
                MTX_FK5EQUAT_F_VSOPFR.dot,
                polarFcartesian)

if __name__ == "__main__":
    None

    
 
 
 
 
 
 
 