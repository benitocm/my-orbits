"""This module contains functions related to the calculation of near parabolic orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
"""
# Standard library imports
from math import isclose, sqrt, isclose
import logging

# Third party imports
import numpy as np
from numpy import cos, sin, arctan
from numba import jit

# Local application imports
from ..util.general import NoConvergenceError, mu_Sun
from ..util.stumpff import calc_stumpff
from ..util.timeut import norm_rad
from ..init_config import NEAR_PARABOLIC_ABS_TOL

logger = logging.getLogger(__name__)

def calc_f(e, c1, c2, c3, u):
    """Compute the true anomaly

    Parameters
    ----------
    e : float
        Eccentricity
    c1 : float
        [description]
    c2 : float
        [description]
    c3 : float
        [description]
    u : float
        [description]

    Returns
    -------
    float
        The mean anomlay [radians]
    """
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
    t_mjd : float
        Time of the computation in Julian centuries since J2000
    mu : float, optional
        Gravitational parameter [AU^3/days^2]                
    abs_tol : float, optional
        Absolute tolerance for the root calculation
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
    Raises
    ------
    NoConvergenceError
        When the method to find the root of the Kepler's equation does not converge
            
    """
    E_2 = 0.0    
    factor = 0.5 * e
    cte = sqrt(mu/(q*(1.0+e)))
    tau = sqrt(mu)*(t_mjd-tp_mjd)
    for _ in range(max_iters):
        E20 = E_2 
        A = 1.5 * sqrt(factor/q)*tau/q
        B = np.cbrt(sqrt(A*A+1.0)+A)
        u = B - 1.0/B 
        u_2 = u*u
        E_2 = u_2*(1.0-e)/factor 
        c1, c2, c3 = calc_stumpff(E_2)
        factor = 3.0*e*c3 
        if np.abs(E_2-E20) < abs_tol :
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
