"""
This module contains functions related to orbit calculations
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

from datetime import datetime
from functools import partial
from collections import namedtuple
import re as re

# Third party imports
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan, arctan2, deg2rad, rad2deg
import toolz as tz
from toolz import pipe, compose

# Local application imports
from myorbit.util import time as tc
from myorbit.util.time import sin_dgms, cos_dgms, tan_dgms, PI_HALF, PI, TWOPI

import logging
logger = logging.getLogger(__name__)


# Horizon Coordinate system
HORIZ_TYPE= 'horiz'
# Equatorial 1 Coordinate system (hour_angle,dec)
EQUAT1_TYPE='equat1'  
# Equatorial 2 Coordinate system (ra,dec)
EQUAT2_TYPE='equat2'  
# Ecliptic Coordinate System
ECLIP_TYPE='eclip' 


def Rx_3d(theta):
    """ 
    Computes the 3x3 rotation matrix to rotate an angle
    into the x axis
    Args:
        theta : angle to rotate [rads]
        
    Returns :
        a 3x3 matrix 
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])

def Ry_3d(theta):
    """ 
    Computes the 3x3 rotation matrix to rotate an angle
    into the y axis
    Args:
        theta : angle to rotate [rads]
        
    Returns :
        a 3x3 matrix 
    """

    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c,0,-s],[0,1,0],[s,0,c]])

def Rz_3d(theta):
    """ 
    Computes the 3x3 rotation matrix to rotate an angle
    into the z axis
    Args:
        theta : angle to rotate [rads]
        
    Returns :
        a 3x3 matrix 
    """    
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])


def mtx_gauss_vectors (Omega, i, omega):
    """ 
    Computes the transformation matrix from the orbital plane coordinate
    system to the ecliptic system being the sun at the center

    Its column vectors (P,Q,R) are the Gauss vectors

    It is equivalent and the U matrix:
          Rz_3d(-Omega).dot(Rx_3d(-i)).dot(Rz_3d(-omega))

    Args:
        Omega : The longitude of the ascending node, angle between 
                the vernal equinox and the point at which the comet
                crosses the ecliptic [rads]
        i : Inclination, angle of intersection between the orbital plane
            and the ecliptic. >90 means the comet is retrograde, its
            directionof revolution around the Sun begin opposite to that
            of the planets [rads].
        omega : Argument of perihelion is the angle between the direction of the
                ascending node an dthe direction of the closest point of the
                orbit ot the Sun [rads].
        
    Returns :
        A 3x3 matrix to th 
    """    
    return Rz_3d(-Omega).dot(Rx_3d(-i)).dot(Rz_3d(-omega))


def cartesianFpolar(r_phi_theta):
    """ 
    Computes the cartesian coordinates from the polar

    Args:
        r_phi_theta : a numpy vector with 3 positions where 
                    the 0 index is the modulus, 
                    the 1 index (theta) angle from x-axis [0,2pi] [rads]
                    the 2 index (phi) is angle from z-axis [0,pi] [rads]
    Returns :
        A 3-vector numpy vector where:
            the 0 index is x
            the 1 index is y
            the 2 index is z
    """    

    r = r_phi_theta[0]
    phi = r_phi_theta[1]
    theta = r_phi_theta[2]
    sin_phi = sin(phi)
    cos_phi = cos(phi)
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    return r*np.array([cos_theta*cos_phi, cos_theta*sin_phi, sin_theta])
	
	
def polarFcartesianv2(xyz):    
    """ 
    Computes the polar coordinates from the cartesian

    Args:
        xyz : a numpy vector with 3 positions where 
            the 0 index is x
            the 1 index is y
            the 2 index is z
    Returns :
        A 3-vector numpy vector where:
                the 0 index is the modulus, 
                the 1 index (theta) angle from x-axis [0,2pi] [rads]
                the 2 index (phi) is angle from z-axis [0,pi] [rads]
    """    
    r = np.linalg.norm(xyz)
    phi = arctan2(xyz[1],xyz[0]) if xyz[0] != 0 else 0
    if phi < 0 :
        phi += 2*np.pi
    #phi = arctan2(np.linalg.norm(xyz[0:2]),xyz[2]) if xyz[2] != 0 else 0
    theta = arctan2(xyz[2],np.linalg.norm(xyz[0:2])) if xyz[2] != 0 else 0
    return np.array([r,phi,theta])	

def polarFcartesian(xyz):    
    """ 
    Computes the polar coordinates from the cartesian

    Args:
        xyz : a numpy vector with 3 positions where 
            the 0 index is x
            the 1 index is y
            the 2 index is z
    Returns :
        A 3-vector numpy vector where:
                the 0 index is the modulus, 
                the 1 index (theta) angle from x-axis [0,2pi] [rads]
                the 2 index (phi) is angle from z-axis [0,pi] [rads]
    """    
    r = np.linalg.norm(xyz)
    if (xyz[0] == 0.0) and (xyz[1] == 0.0) :
        phi = 0.0
    else :
        phi = arctan2(xyz[1],xyz[0]) 
    if phi < 0 :
        phi += 2*np.pi
    rho = np.linalg.norm(xyz[0:2])
    if (xyz[2] == 0.0) and (rho == 0.0) :
        theta = 0.0
    else :
        theta = arctan2(xyz[2],rho) 
    return np.array([r,phi,theta])	



# Polynomials for the precession

a1_pol = partial(tc.polynomial, 2306.2181, 1.39656, -0.000139, 0, 0)
a2_pol = partial(tc.polynomial, 0.30188, -0.000344, 0, 0, 0)

b2_pol = partial(tc.polynomial, 1.09468, 0.000066 ,0, 0, 0)

c1_pol = partial(tc.polynomial, 2004.3109, -0.85330, 0.000217, 0, 0)
c2_pol = partial(tc.polynomial, 0.42665, .000217 ,0 , 0, 0)

def prec_mtx_equat(T1,T2) : 
    """ 
    Computes the precession of matrix for equatorial coordinates

    Args:
        T1: Epoch given in Julian centuries from J2000 (result of calling T() with year,month, and frac day of the epoch)
        T2: Epoch to precess to in Julian centuries since J2000 (result of calling T() with year,month, and frac day of the epoch)

    Returns :
        A 3x3 matrix 
    """
    # interval in Julian centuries between the epoch
    # to precess and the epoch given
    dT = T2-T1 
    
    a4 = b4 = c4 = 0 

    a3 = 0.017998
    zeta = pipe(tc.polynomial(0,a1_pol(T1),a2_pol(T1),a3,a4,dT),tc.sec2rads)
    
    b3 = 0.018203
    z = pipe(tc.polynomial(0,a1_pol(T1),b2_pol(T1),b3,b4,dT),tc.sec2rads)
    
    c3 = -0.041833
    theta = pipe(tc.polynomial(0,c1_pol(T1),-c2_pol(T1),c3,c4,dT),tc.sec2rads)
    
    return Rz_3d(-z).dot(Ry_3d(theta)).dot(Rz_3d(-zeta))


e0_pol = partial(tc.polynomial, 174.876383889*3600, 3289.4789, 0.60622, 0, 0)
e1_pol = partial(tc.polynomial, 869.8089, 0.50491, 0, 0, 0)

f1_pol = partial(tc.polynomial, 5029.0966, 2.22226, -0.000042, 0, 0)
f2_pol = partial(tc.polynomial, 1.11113, -0.000042, 0, 0, 0)

g1_pol = partial(tc.polynomial, 47.0029, -0.06603, 0.000598, 0, 0)
g2_pol = partial(tc.polynomial, -0.03302, 0.000598, 0, 0, 0)


def n_PPI_p (T1,T2) :
    dT = T2-T1    

    PPI = pipe(tc.polynomial(e0_pol(T1), -e1_pol(T1), 0.03536, 0, 0, dT),tc.sec2rads)

    p = pipe(tc.polynomial(0, f1_pol(T1), f2_pol(T1), -0.000006, 0, dT),tc.sec2rads)

    n = pipe(tc.polynomial(0, g1_pol(T1), g2_pol(T1), 0.000060, 0, dT),tc.sec2rads)

    return n, PPI, p



def mtx_eclip_prec(T1,T2) : 
    """ 
    Computes the precession of matrix for ecliptic coordinates

    Args:
        T1: Epoch given in Julian centuries from J2000 (result of calling T() with year,month, and frac day of the epoch)
        T2: Epoch to precess to in Julian centuries since J2000 (result of calling T() with year,month, and frac day of the epoch)

    Returns :
        A 3x3 matrix
    """

    #T1 is zero is the initial epoch is J2000

    # interval in Julian centuries between the epoch
    # to precess and the epoch given
    dT = T2-T1    

    PPI = pipe(tc.polynomial(e0_pol(T1), -e1_pol(T1), 0.03536, 0, 0, dT),tc.sec2rads)

    p = pipe(tc.polynomial(0, f1_pol(T1), f2_pol(T1), -0.000006, 0, dT),tc.sec2rads)

    ppi = pipe(tc.polynomial(0, g1_pol(T1), g2_pol(T1), 0.000060, 0, dT),tc.sec2rads)

    return Rz_3d(-PPI-p).dot(Rx_3d(ppi)).dot(Rz_3d(PPI))

def prec_mtx_eclipv2(T1,T2) : 
    """ 
    Computes the precession of matrix for ecliptic coordinates

    Args:
        T1: Epoch given in Julian centuries from J2000 (result of calling T() with year,month, and frac day of the epoch)
        T2: Epoch to precess to in Julian centuries since J2000 (result of calling T() with year,month, and frac day of the epoch)

    Returns :
        A 3x3 matrix
    """

    #T1 is zero is the initial epoch is J2000

    # interval in Julian centuries between the epoch
    # to precess and the epoch given
    dT = T2-T1    
    Rad = np.pi/180.0
    Arcs = 3600.0*180.0/np.pi

    Pi = 174.876383889 * Rad + (((3289.4789 + 0.60622 * T1) * T1) +((-869.8089 - 0.50491 * T1) + 0.03536 * dT) * dT) / Arcs
    pi = ((47.0029 - (0.06603 - 0.000598 * T1) * T1) +((-0.03302 + 0.000598 * T1) + 0.000060 * dT) * dT) * dT / Arcs
    p_a = ((5029.0966 + (2.22226 - 0.000042 * T1) * T1) +((1.11113 - 0.000042 * T1) - 0.000006 * dT) * dT) * dT / Arcs    
    return Rz_3d(-(Pi + p_a)).dot(Rx_3d(pi)).dot(Rz_3d(Pi))

    
def format_time(hhmmss,as_int=False):
    if (as_int):
        return '{:02.0f}h{:02.0f}m{:02.0f}s'.format(*hhmmss)
    else :
        return '{:02.0f}h{:02.0f}m{:06.3f}s'.format(*hhmmss)

def format_dg(dg,m,s,sign):
    out_str= "{:02.0f}°{:02.0f}'{:02.0f}\"".format(dg,m,s)
    return '+'+out_str if sign>0 else '-'+out_str


LAT_PATTERN_1 = re.compile(r"(\+|-)?(\d*)[°d](\d*)m([\d\.]*)s\s*(N|S)?")
def make_lat(lat_str : str) -> float :
    match= LAT_PATTERN_1.match(lat_str)
    if match :
        sign = 1
        if match.groups()[0] is not None:
            sign = 1 if match.groups()[0]=='+' else -1
        dg = float(match.groups()[1])
        m = float(match.groups()[2])
        s = float(match.groups()[3])
        if match.groups()[4] is not None:
            sign = 1 if match.groups()[4]=='N' else -1
        return pipe(tc.dgms2dg(dg,m,s,sign),deg2rad)
    else :
        return 0


LON_PATTERN_1 = re.compile(r"(\+|-)?(\d*)[°d](\d*)m([\d\.]*)s\s*(W|E)?")
def make_lon(lon_str:str) -> float:
    """ 
    Create a longitude in radians from a string value in degrees

    Args:
        lon_str: Longitude in angle units as string (e.g. "0d43m35.5s")
                 [0,360]

    Returns :
        A longitude value in radians
    """    
    match= LON_PATTERN_1.match(lon_str)
    if match :
        sign = 1
        if match.groups()[0] is not None:
            sign = 1 if match.groups()[0]=='+' else -1
        dg = float(match.groups()[1])
        m = float(match.groups()[2])
        s = float(match.groups()[3])
        if match.groups()[4] is not None:
            sign = 1 if match.groups()[3]=='E' else -1
        return pipe(tc.dgms2dg(dg,m,s,sign),deg2rad)
    else :
        return 0         

RA_PATTERN_1 = re.compile(r"(\d*)h(\d*)m([\d\.]*)s")
def make_ra(ra_str):
    match= RA_PATTERN_1.match(ra_str)
    if match :
        h = float(match.groups()[0])
        m = float(match.groups()[1])
        s = float(match.groups()[2])        
        return pipe(tc.hms2dg(h,m,s),deg2rad)
    else :
        return 0

Coord = namedtuple('Coord',['v','equinox','type'])
"""
Named tuple to build a celestial coordinate in radians
"""

def mk_co_equat2(ra_str, lat_str,equinox='',r=1):
    """ 
    Create a coordinate in the Equatorial-2 coordinate system. By default,
    in J2000 equinox

    Args:
        ra_str : Right ascension in time units as string (e.g. "12h18m47.5s")
                 [0,24h)
        dec_str: Declination in angle units as string (e.g. "-0d43m35.5s")
                 [-90,90]

    Returns :
        A Equatorial 2 coordinate (Coord object) 
    """    
    lon = make_ra(ra_str)
    lat = make_lat(lat_str)
    return Coord(np.array([r,lon,lat]),equinox, EQUAT2_TYPE)


def mk_co_equat1(ha_str, dec_str,equinox=''):
    """ 
    Create a coordinate in the Equatorial-1 coordinate system (hour angle and declination). By default,
    in J2000 equinox 

    Args:
        ha_str : Hour Angle in time units as string (e.g. "12h18m47.5s")
                 [0,24h)
        dec_str: Declination in angle units as string (e.g. "-0d43m35.5s")
                 [-90,90]

    Returns :
        A Equatorial 1 coordinate (Coord object) 
    """    
    ha = make_ra(ha_str) # similar as ra although the interpretation is different
    lat = make_lat(dec_str)
    return Coord(np.array([1,ha,lat]),equinox, EQUAT1_TYPE)

def mk_co_eclip (lambda_str, beta_str,equinox=''):
    """ 
    Create a coordinate in the Ecliptic coordinate system. By default,
    in J2000 equinox

    Args:
        lambda_str : Ecliptic longitude in angle units as string (e.g. "184d43m35.5s")
                     [0,360) degrees
        beta_str: Ecliptic latitude  in angle units as string (e.g. "-15d43m35.2s")
                 [-90,90] degrees

    Returns :
        An Ecliptic coordinate (Coord object) with components in radians
    """    
    lon = make_lon(lambda_str)
    lat = make_lat(beta_str)
    return Coord(np.array([1,lon,lat]),equinox,ECLIP_TYPE)

def mk_co_horiz (az_str, alt_str, equinox=''):
    """ 
    Create a coordinate in the Horizontal coordinate system.

    Args:
        az_str : Azimut in angle units as string (e.g. "184d43m35.5s")
                 [0,360) degrees.
                 Assumes its origin in the south as Meeus recommend. If not,
                 the adjust can be applied
                 
        alt_str: Altitude in angle units as string (e.g. "84d43m35.5s")
                 [-90,90] degrees
                 
    Returns :
        An Horizontal coordinate (Coord object) with components in radians
    """    
    az = make_lon(az_str)
    #if adjust is not None:
    #    az = az - adjust
    alt = make_lat(alt_str)
    return Coord(np.array([1,az,alt]),equinox,HORIZ_TYPE)

obliq_pol = partial(tc.polynomial,23.43929111*3600, -46.815, -0.00059, 0.001813,0)
#obliq_pol = partial(tc.polynomial,23.43929111,-46.815/3600,-0.00059/3600.0,0.001813/3600.0,0)

def p_equat(v):
    c = Coord(v,'', EQUAT2_TYPE)    
    print (as_str(c))


def obliquity (T : float) -> float :
    """
    Returns the obliquity angle of the ecliptic [rads]
    """
    return pipe(obliq_pol(T)/3600.0,deg2rad)

"""
def mtx_eclipFequat(epoch_name="J2000"):
    eps = pipe(tc.T(epoch_name),obliquity)
    return Rx_3d(eps)

def mtx_equatFeclip(epoch_name="J2000"):
    eps = pipe(tc.T(epoch_name),obliquity)
    return Rx_3d(-eps)
"""

def mtx_equatFeclip(T):
    """ 
    Build the matrix to transform from Eclipti to Equatorial.

    Args:
        T : Time in Julian centuries since J2000
                 
    Returns :
        A numpy  3x3 matrix
    """    

    eps = pipe(T,obliquity)
    return Rx_3d(-eps)

def mtx_eclipFequat(T):
    """ 
    Build the matrix to transform from Equatorial to Ecliptic.

    Args:
        T : Time in Julian centuries since J2000
                 
    Returns :
        A numpy  3x3 matrix
    """    

    eps = pipe(T,obliquity)
    return Rx_3d(eps)



def eclipFequat(equat, epoch_name="J2000"):
    mtx = mtx_eclipFequat(epoch_name)
    r_lambda_beta = pipe(equat.v,cartesianFpolar,mtx.dot,polarFcartesian)
    return Coord(r_lambda_beta,epoch_name,ECLIP_TYPE)


def equat2Feclip(eclip, epoch_name="J2000"):
    mtx = mtx_equatFeclip(epoch_name)
    r_ra_dec = pipe(eclip.v,cartesianFpolar,mtx.dot,polarFcartesian)
    return Coord(r_ra_dec,epoch_name,EQUAT2_TYPE)


def horizFequat1(equat,lat,adjust=PI):
    """
    Computes AltAzimut coordinate from the equatorial (ha, dec)

    Args:
        equat: An equatorial1 Coord object 
        lat  : latitude in radians 
    
    Returns:
        AltAzmiut Coord object        
    """
    mtx = Ry_3d(PI_HALF-lat)
    r_az_alt = pipe(equat.v,cartesianFpolar,mtx.dot,polarFcartesian)
    r_az_alt[1] = r_az_alt[1] + adjust
    return Coord(r_az_alt,"",HORIZ_TYPE)


def equat1Fhoriz(horiz,lat):
    """
    Computes AltAzimut coordinate from the equatorial (ha, dec)

    Args:
        horiz: An horizontal Coord object 
        lat  : latitude in radians 
    
    Returns:
        AltAzmiut Coord object        
    """

    mtx = Ry_3d(-1.0*(PI_HALF-lat))
    r_ha_dec = pipe(horiz.v,cartesianFpolar,mtx.dot,polarFcartesian)
    return Coord(r_ha_dec,"",EQUAT1_TYPE)


def lat_as_str(lat_rad):
    dgms = pipe(lat_rad,rad2deg,tc.dg2dgms)    
    return format_dg(*dgms)

def lon_as_str(lon_rad):
    return lat_as_str(lon_rad)


def as_str(c):    
    if c.type == EQUAT1_TYPE:        
        hms = pipe(c.v[1],rad2deg,tc.dg2h,tc.h2hms)
        hms = format_time(hms)
        dgms = pipe(c.v[2],rad2deg,tc.dg2dgms)
        dgms = format_dg(*dgms)
        return f"EQUAT1: HA {hms}  DEC {dgms}"
    elif c.type == EQUAT2_TYPE:        
        hms = pipe(c.v[1],rad2deg,tc.dg2h,tc.h2hms)
        hms = format_time(hms)
        dgms = pipe(c.v[2],rad2deg,tc.dg2dgms)
        dgms = format_dg(*dgms)
        return f"EQUAT2: RA {hms}  DEC {dgms}"
    elif c.type == ECLIP_TYPE:       
       dgms_l = pipe(c.v[1],rad2deg,tc.dg2dgms)
       dgms_l = format_dg(*dgms_l)
       dgms_b = pipe(c.v[2],rad2deg,tc.dg2dgms)
       dgms_b = format_dg(*dgms_b)
       return f"ECLIP: L {dgms_l}  B {dgms_b}"
    elif c.type == HORIZ_TYPE:
       dgms_az = pipe(c.v[1],rad2deg,tc.dg2dgms)
       dgms_az = format_dg(*dgms_az)
       dgms_alt = pipe(c.v[2],rad2deg,tc.dg2dgms)
       dgms_alt = format_dg(*dgms_alt)
       return f"HORIZ: AZ {dgms_az}  ALT {dgms_alt}"
    else :
        return f"Unknown format {c.type}"




def change_equinox_equat(from_epoch:str ,to_epoch:str ,equat:Coord) -> Coord :
    """
    Transform the equatorial coordinate from one equinox to other

    Args:
        T1 : Julian centuries of the epoch given
        T2 : Julian centuries of the epoch to precess

    """    
    mtx = prec_mtx_equat(tc.T(from_epoch),tc.T(to_epoch))    
    r_ra_dec = pipe(equat.v,cartesianFpolar,mtx.dot,polarFcartesian)
    return Coord(r_ra_dec,to_epoch,EQUAT2_TYPE)

def change_equinox_eclip(from_epoch:str, to_epoch:str, eclip:Coord) -> Coord :
    """
    Transform the eclipt coordinate from one equinox to other

    Args:
        from_epoch : Julian centuries of the epoch given
        to_epoch : Julian centuries of the epoch to precess


    """    
    mtx = mtx_eclip_prec(tc.T(from_epoch),tc.T(to_epoch))    
    r_lambda_beta = pipe(eclip.v,cartesianFpolar,mtx.dot,polarFcartesian)
    return Coord(r_lambda_beta,to_epoch,ECLIP_TYPE)


if __name__ == "__main__":

    import sys

    
    #sys.exit(0)


    #equat1 = mk_co_equat2("12h18m47.5s","-0d43m35.5s")
    #print (as_str2(equat1))
    #eclip1 = eclipFequatv2(equat1)
    #print (as_str2(eclip1))

    #print ("..............")
    #eclip2 = mk_co_eclip("184°36m0s","01°12m0s")
    #print (as_str2(eclip2))
    #equat2 = equat2Feclipv2(eclip2)
    #print (as_str2(equat2))

    
    #equat1 = mk_co_equat1("16h29m45s","-0d30m30s")
    #lat_obs = make_lat("25d0m0sN")
    #print (as_str2(equat1))
    #print (lat_as_str(lat_obs))
    #horiz1 = horizFequat1(equat1,lat_obs)
    #print (as_str2(horiz1))


    #print (co.equat2horiz(rad2deg(lat_obs),tc.hms2h(16,29,45),tc.dgms2dg(0,30,30,-1)))

    #horiz = mk_co_horiz("115°0m0s","40°0m0s",adjust=PI)
    #print (as_str2(horiz))
    #lat_obs = make_lat("38°0m0sN")
    #print (lat_as_str(lat_obs))
    #equat = equat1Fhoriz(horiz,lat_obs)
    #print (horiz)
    #print (equat)
    #print (as_str2(equat))

    #kk = mk_co_equat1("1h46m14.676s","-40°34m53.36s",'')
    #print (kk)

    #lat_obs = make_lat("35°36m5.115sN")
    #horiz = mk_co_horiz("200°10m20s","10°0m0s",adjust=PI)
    #equat = equat1Fhoriz(horiz,lat_obs)
    #print (as_str2(equat))


    #equat = mk_co_equat2("11h10m13s","30d05m40s")
    #eclip = eclipFequatv2(equat)
    #exp_eclip = mk_co_eclip("157°19m09s","22°41m54s","J2000")
    #print (eclip.v)
    #print (exp_eclip.v)
    #print (as_str2(eclip))
    #print (as_str2(exp_eclip))


    #equat1 = mk_co_equat2("12h18m47.5s","-0d43m35.5s")
    #print (as_str2(equat1))
    #eclip1 = eclipFequatv2(equat1)
    #print (as_str2(eclip1))
    #print (make_lon("-2°0m0s"))



    #print (as_str2(equat))
    #T1 = 0
    #T2 = 0
    #dT = 0.288670500
    #precession_mtx_equat(T1,T2,dT) 
    #
    # print (T("J2000"))
    # J200 correspond to tc.datefd2jd(2000,1,1.5)
    #print  (tc.t)

    # The initial epoch is J2000 -> tc.datefd2jd(2000,1,1.5)
    # The final epoch is 2028 November 13.18 --> tc.datefd2jd(2028,11,13.18)
    #J2000 = tc.datefd2jd(2000,1,1.5)
    #T1 = T(2000,1,1.5)
    #T2 = T(2028,11,13.18)

    f_epoch="J2000.0"
    t_epoch="2028.11.13.19"
    equat_1 = mk_co_equat2("2h44m12.975s","49°13m39.90s")
    equat_2 = change_equinox_equat(f_epoch,t_epoch,equat_1)
    print (as_str(equat_2))

    #mtx = prec_mtx_equat(T1,T2)
    #equat_1 = mk_co_equat2("2h44m12.975s","49°13m39.90s")
    #r_ra_dec = pipe(equat_1.v,cartesianFpolar,mtx.dot,polarFcartesian)
    #equat_2 = Coord(r_ra_dec,"",EQUAT2_TYPE)
    #print (as_str2(equat_1))
    #print (as_str2(equat_2))
    #epoch_tup = (2000,1,1.5)
    #epoch_tup = "2010.0"
    #eps = pipe(T(epoch_tup),obliquity)
    #print (rad2deg(eps))

    
    f_epoch="J2000.0"
    t_epoch="-214.30.0"
    eclip_1 = mk_co_eclip("149°28m54.984s","1°45m55.76s")
    eclip_2 = change_equinox_eclip(f_epoch,t_epoch,eclip_1)
    print (as_str(eclip_1))
    print (as_str(eclip_2))


    f_epoch="1950.0"
    t_epoch="J2000.0"
    equat_1 = mk_co_equat2("18h53m48s","43°53m0s")
    equat_2 = change_equinox_equat(f_epoch,t_epoch,equat_1)
    print (as_str(equat_1))
    print (as_str(equat_2))

    print ("\n\n")
    f_epoch="1950.0"
    t_epoch="2010.0"
    #equat_1 = mk_co_equat2("11h10m13s","30°05m40s")    
    #equat_2 = change_equinox_equat(f_epoch,t_epoch,equat_1)
    #equat_3 = change_equinox_equat(t_epoch,f_epoch,equat_2)

    eclip_1 = mk_co_eclip("149°28m54.984s","1°45m55.76s")
    eclip_2 = change_equinox_eclip(f_epoch,t_epoch,eclip_1)
    eclip_3 = change_equinox_eclip(t_epoch,f_epoch,eclip_2)
    print (as_str(eclip_1))
    print (as_str(eclip_2))
    print (as_str(eclip_3))

    print ("\n\n")
    f_epoch="1950.0"
    t_epoch="2010.0"
    eclip_1950 = mk_co_eclip("149°28m54.984s","1°45m55.76s")
    equat_1950 = equat2Feclip(eclip_1950,f_epoch)
    equat_2000 = change_equinox_equat(f_epoch,t_epoch,equat_1950)
    eclip_2000 = eclipFequat(equat_2000)
    eclip_2000_1 = change_equinox_eclip(f_epoch,t_epoch,eclip_1950)
    print (as_str(eclip_2000))
    ##print (as_str(eclip_2))
    print (as_str(eclip_2000_1))










    #eclip_1 = eclipFequat(equat_2)
    #eclip_2 = change_equinox_equat(t_epoch,f_epoch,eclip_1)
    #equat_3 = equat2Feclip(eclip_2)
    #print (as_str(equat_1))
    #print (as_str(eclip_1))
    #print (as_str(equat_2))
    #print (as_str(eclip_2))
    #print (as_str(equat_3))






    







    



    
