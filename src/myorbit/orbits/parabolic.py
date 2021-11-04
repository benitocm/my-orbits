"""This module contains functions related to the calculation of Keplerian orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
"""
# Standard library imports
from math import isclose, sqrt
import logging
from math import isclose

# Third party imports
import numpy as np
from numpy import cos, sin, arctan

# Local application imports
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
    """Computes Mean Anomaly as function of time 

    Parameters
    ----------
    t_mjd : float
        Time of the computation (Modified Julian day)
    tp_mjd : float
        Time of periapse (Modified Julian day)
    a : float
        Semimajor axis [AU]

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
    tuple (r_xyz, rdot_xyz, r, h, Mp, f, E):
        r_xyz: is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane (perifocal frame) [AU]
        rdot_xyz: is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane (perifocal frame) [AU/days]
        r : Modulus of the radio vector of the object (r_xyz) but calculated following the polar equation [AU]    
        h : Angular momentum (deduced from geometic properties)
        Mp : Mean anomaly at time of computation [radians]
        f : True anomaly at time of computation [radians]
        E : None
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

#
# An alternative approach using Stumpff functions 
# according to the book
#       "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
# 
  
def _calc_stumpff_values(E_2, epsilon=1e-7, max_iters=100):    
    """Computes the values for the Stumpff functions C1, C2, C3 following
    an iterative procedure

    Parameters
    ----------
    E_2 : float
        Square of eccentric anomaly [radians^2]
    epsilon : float, optional
        Relative accuracy, by default 1e-7
    max_iters : int, optional
        Maximum number of iterations, by default 100

    Returns
    -------
    tuple (C1, C2, C3)
        The three stumpff values [float]
    """

    c1, c2, c3 = 0.0, 0.0, 0.0
    to_add = 1.0
    for n in range(1 ,max_iters):
        c1 += to_add
        to_add /= (2.0*n)
        c2 += to_add
        to_add /= (2.0*n+1.0)
        c3 += to_add
        to_add *= -E_2
        if isclose(to_add, 0, abs_tol=epsilon) :
        #if np.isclose(to_add,epsilon,atol=epsilon):
            return c1, c2, c3
    logger.error(f"Not converged after {n} iterations")
    return c1, c2, c3

def calc_rv_for_parabolic_orbit_stumpff (tp_mjd, q, e, t_mjd, max_iters=30):
    """Computes the position (r) and velocity (v) vectors for parabolic orbits using
    an iterative method (Newton's method) for solving the Kepler equation.
    (pg 66 of Astronomy on the Personal Computer book). The m_anomaly is
    the one that varies with time so the result of this function will also vary with time.

    Parameters
    ----------
    tp : float
        Time of perihelion passage in Julian centuries since J2000
    q : float
        Perihelion distance [AU]
    e : float
        Eccentricity of the orbit
    t : float
        Time of the computation in Julian centuries since J2000
    max_iters : int, optional
        Maximum number of iterations, by default 15

    Returns
    -------
        tuple (r_xyz, rdot_xyz, M, f, E, h, r):
            r_xyz: is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
                with respect to the orbital plane (perifocal frame) [AU]
            rdot_xyz: is a np.array[3] that contains the velocity vector (cartesian) of the body
                with respect to the orbital plane (perifocal frame) [AU/days]
            r : Modulus of the radio vector of the object (r_xyz) but calculated following the polar equation [AU]    
            h : Angular momentum (deduced from geometic properties)
    """
    E_2 = 0.0    
    factor = 0.5 * e
    cte = sqrt(GM/(q*(1.0+e)))
    tau = sqrt(GM)*(t_mjd-tp_mjd)
    EPSILON = 1e-7

    for i in range(max_iters):
        E20 = E_2 
        A = 1.5 * sqrt(factor/(q*q*q))*tau
        B = np.cbrt(sqrt(A*A+1.0)+A)
        u = B - 1.0/B 
        u_2 = u*u
        E_2 = u_2*(1.0-e)/factor 
        c1, c2, c3 = _calc_stumpff_values(E_2)
        factor = 3.0*e*c3 
        if isclose(E_2, E20, abs_tol=EPSILON) :
            R = q * (1.0 + u_2*c2*e/factor)
            r_xyz = np.array([q*(1.0-u_2*c2/factor), q*sqrt((1.0+e)/factor)*u*c1,0.0])
            rdot_xyz = np.array([-cte*r_xyz[1]/R, cte*(r_xyz[0]/R+e),0.0])
            return r_xyz, rdot_xyz, np.linalg.norm(r_xyz), np.linalg.norm(np.cross(r_xyz, rdot_xyz))
    #logger.warning(f"Not converged after {i} iterations")
    logger.error(f'Not converged with q:{q},  e:{e}, t:{t_mjd}, t0:{tp_mjd} after {i} iterations')
    return np.array([0,0,0]), np.array([0,0,0]), 0, np.array([0,0,0])


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

if __name__ == "__main__" :
    test_comet()
    