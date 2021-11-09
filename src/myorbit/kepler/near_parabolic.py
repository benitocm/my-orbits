"""This module contains functions related to the calculation of Keplerian orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
"""
# Standard library imports
from math import isclose, sqrt, isclose
import logging

# Third party imports
import numpy as np
from numpy import cos, sin, arctan

# Local application imports
from myorbit.util.general import pow, NoConvergenceError
from myorbit.util.timeut import norm_rad
from myorbit.util.general import mu_Sun

from pathlib import Path
CONFIG_INI=Path(__file__).resolve().parents[3].joinpath('conf','config.ini')
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read(CONFIG_INI)
NEAR_PARABOLIC_ABS_TOL = float(cfg.get('general','near_parabollic_abs_tol'))


logger = logging.getLogger(__name__)


#
# An alternative approach using Stumpff functions 
# according to the book
#       "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
# 
  
def calc_stumpff_as_series(E_2, epsilon=NEAR_PARABOLIC_ABS_TOL, max_iters=100):    
    """Computes the values for the Stumpff functions C1, C2, C3 summing up
    a infinite series

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
    for n in range(1, max_iters):
        c1 += to_add
        to_add /= (2.0*n)
        c2 += to_add
        to_add /= (2.0*n+1.0)
        c3 += to_add
        to_add *= -E_2
        if isclose(to_add, 0, abs_tol=epsilon) :
            return c1, c2, c3
    
    logger.error(f"Not converged after {n} iterations")
    raise NoConvergenceError((c1,c2,c3), n, n, "Stumpff functions does not converge")

def calc_stumpff_exact(E_2):    
    """Computes the values for the Stumpff functions C1, C2, C3 according to
    its definition

    I have seen that for value of E=

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
    E = np.sqrt(E_2)
    c1 = np.sin(E)/E
    c2 = (1-np.cos(E))/pow(E,2)
    c3 = (E-np.sin(E))/pow(E,3)

    return c1, c2, c3

def calc_f(e, c1, c2, c3, u):
    tanf_div2 = np.sqrt((1+e)/(3*e*c3))*c2*u/c1
    return norm_rad(2*np.arctan(tanf_div2))

def calc_rv_by_stumpff (tp_mjd, q, e, t_mjd, mu=mu_Sun, abs_tol=NEAR_PARABOLIC_ABS_TOL, max_iters=30):
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
            h_xyz : Angular momentum (deduced from geometic properties)
    """
    E_2 = 0.0    
    factor = 0.5 * e
    cte = sqrt(mu/(q*(1.0+e)))
    tau = sqrt(mu)*(t_mjd-tp_mjd)
    for _ in range(max_iters):
        E20 = E_2 
        A = 1.5 * sqrt(factor/(q*q*q))*tau
        B = np.cbrt(sqrt(A*A+1.0)+A)
        u = B - 1.0/B 
        u_2 = u*u
        E_2 = u_2*(1.0-e)/factor 
        c1, c2, c3 = calc_stumpff_exact(E_2)
        factor = 3.0*e*c3 
        if isclose(E_2, E20, rel_tol=0, abs_tol=abs_tol) :
            R = q * (1.0 + u_2*c2*e/factor)
            r_xyz = np.array([q*(1.0-u_2*c2/factor), q*sqrt((1.0+e)/factor)*u*c1,0.0])
            rdot_xyz = np.array([-cte*r_xyz[1]/R, cte*(r_xyz[0]/R+e),0.0])
            f = calc_f(e, c1, c2, c3, u)
            return r_xyz, rdot_xyz, np.linalg.norm(r_xyz), np.cross(r_xyz, rdot_xyz), -1000, f, -1000
    msg = f'Near parabolical method not converged for q:{q},  e:{e}, t:{t_mjd}, t0:{tp_mjd}, error:{np.abs(E_2-E20)} after {max_iters} iterations'
    logger.error(msg)
    raise NoConvergenceError(None, max_iters,max_iters, None,message=msg)
    


if __name__ == "__main__" :
    E=6.2831851329912345
    print (E,)
    E1 = norm_rad(E)

    exp_c1, exp_c2, exp_c3 = calc_stumpff_exact(E*E)
    c1,c2,c3 = calc_stumpff_as_series(E*E, epsilon=1e-10)
    print (f'c1={c1}, exp_c1={exp_c1}, {isclose(c1,exp_c1,abs_tol=1e-9)}')
    print (f'c2={c2}, exp_c2={exp_c2}, {isclose(c2,exp_c2,abs_tol=1e-9)}')
    print (f'c3={c3}, exp_c3={exp_c3}, {isclose(c3,exp_c3,abs_tol=1e-9)}')
