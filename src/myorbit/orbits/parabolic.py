"""This module contains functions related to the calculation of Keplerian orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
"""
# Standard library imports
from math import isclose
import logging
from math import isclose

# Third party imports
import numpy as np
from numpy import cos, sin, arctan

# Local application imports
from myorbit.util.timeut import norm_dg
from myorbit.util.general import pow
from myorbit.util.constants import *

logger = logging.getLogger(__name__)

# For an elliptic orbit:
#   (Energy <0) 
#   0 < eccentricity  < 1

# For an hyperbolic orbit:
#   (Energy > 0) 
#   eccentricity > 1


def calc_Mp (q, t_mjd, tp_mjd):
    """Computes Mean Anomaly as function of time [summary]

    Parameters
    ----------
    t_mjd : float
        Time of the computation (Modified Julian day)
    tp_mjd : float
        Time of periapse (Modified Julian day)
    a : float
        Semimajor axis [AU]
    """

    """
    Returns
    -------
    float
        The mean anomaly [radians]
    """    

    M = np.sqrt(GM/(2*pow(q,3)))*(t_mjd-tp_mjd)
    return M


def calc_rv_for_parabolic_orbit (tp_mjd, q, t_mjd):
    """[summary]

    Parameters
    ----------
    t_mjd : [type]
        [description]
    tp_mjd : [type]
        [description]
    a : [type]
        [description]
    e : [type]
        [description]

    Returns
    -------
    tuple (r_xyz, rdot_xyz, M, f, E, h, r):
        r_xyz: is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane [AU]
        rdot_xyz: is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane [AU/days]
        M : Mean anomaly at time of computation [rads]
        f : True anomaly at time of computation [rads]
        E : Eccentric anomaly at time of computation [rads]
        h : Angular momentum (deudced from geometic properties)
        r : Modulus of the radio vector of the object [AU]
    """

    # The Parabolic Mean Anomaly is calculated as a function of time
    Mp = calc_Mp(q, t_mjd, tp_mjd)

    # z, an auxiliary quantity is calculated
    z = np.cbrt(1.5*Mp+(np.sqrt(1+(pow(1.5*Mp,2)))))

    tan_fdiv2 = z - (1/z)

    f = 2*arctan(tan_fdiv2)

    h = np.sqrt(GM*2*q)

    r = 2*q/(1 + np.cos(f))

    r_xyz = np.array([r*cos(f), r*sin(f), 0.0])

    rdot_xyz = np.array([-GM*np.sin(f)/h,GM*(1+np.cos(f))/h , 0.0]) 

    #if not np.allclose(rdot_xyz,r1dot_xyz,atol=1e-012):
    #    print (f'Differences between r1: {rdot_xyz} and r2:{r1dot_xyz}')

    return r_xyz, rdot_xyz, r, h, Mp, f, None
  


def quadrant (alpha) :
    dg = np.rad2deg(alpha)
    if dg < 0:
        dg = norm_dg(dg)
    if 0 <= dg <= 90 :
        return 1
    elif 90 < dg <= 180 :
        return 2
    elif 180 < dg <= 270 :
        return 3
    else :
        return 4 

def test_comet():
    # For comets, we have Perihelion distance (q or rp) instead of semimajor axis (a)
    # For asteroids, we have the semimajor axis (a)    t_mjd = 56197.0
    TP_MJD = 57980.231000000145
    T0_MJD = 57966.0
    q = 2.48315593 # Perihelion distance
    a = None
    e = 1.0
    hs= []
    for dt in range(0,100):
        t_mjd = T0_MJD + dt
        r_xyz, rdot_xyz, Mp, f, r, h_geo = calc_rv_for_parabolic_orbit(t_mjd, TP_MJD, q)
        #print (r_xyz, rdot_xyz, np.rad2deg(Mp), np.rad2deg(f)) 
        h_rv = np.cross(r_xyz,rdot_xyz)
        #print (h_geo, h_rv[2], isclose(h_geo, h_rv[2], abs_tol=1e-12))
        hs.append(h_rv[2])
        #print (np.rad2deg(f),np.rad2deg(Mp))
        #print (quadrant(f), quadrant(Mp))
        #Energy = - GM/(2*a)
        #v = sqrt(2*(Energy+(GM/r)))
        v = np.sqrt(2*GM/r)
        print (v, np.linalg.norm(rdot_xyz), isclose(v,np.linalg.norm(rdot_xyz),abs_tol=1e-12))
    is_h_cte = all(isclose(h, hs[0], abs_tol=1e-12) for h in hs)
    print (is_h_cte)

def test_body():
     None   


    




if __name__ == "__main__" :
    test_comet()
    