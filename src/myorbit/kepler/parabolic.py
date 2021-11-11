"""This module contains functions related to the calculation of Keplerian orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
"""
# Standard library imports
from math import isclose
import logging

# Third party imports
import numpy as np
from numpy import cos, sin, arctan
from myorbit.util.constants import TWOPI
from numba import jit

# Local application imports
from myorbit.util.general import pow, NoConvergenceError
from myorbit.util.timeut import norm_rad
from myorbit.util.general import mu_Sun

logger = logging.getLogger(__name__)

# For an elliptic orbit:
#   (Energy <0) 
#   0 < eccentricity  < 1

# For an hyperbolic orbit:
#   (Energy > 0) 
#   eccentricity > 1

# For a parabolic orbit:
#   (Energy = 0) 
#   eccentricity = 1

# For this type of orbit, I have also followed:
#    https://en.wikipedia.org/wiki/Parabolic_trajectory


def calc_Mp (q, t_mjd, tp_mjd, mu=mu_Sun):
    """Computes the parabolic Mean Anomaly as function of time 

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
        The parabolic mean anomaly [radians]
    """    
    return (np.sqrt(mu/(2*q))/q)*(t_mjd-tp_mjd)
    

def calc_rv_for_parabolic_orbit (tp_mjd, q, t_mjd, mu=mu_Sun):
    """[summary]

    Parameters
    ----------
    tp_mjd : float
        Time of perihelion passage [Modified Julian day]
    q : float
        Distance to the perihelion [AU]
    t_mjd : float
        Time of the computation [Modified Julian day]

    Returns
    -------
    tuple (r_xyz, rdot_xyz, r, h_xyz, Mp, f, None):
        r_xyz: is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane (perifocal frame) [AU]
        rdot_xyz: is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane (perifocal frame) [AU/days]
        r : Modulus of the radio vector of the object (r_xyz) but calculated following the polar equation [AU]    
        h_xyz : Angular momentum (deduced from geometic properties)
        Mp : Parabolic mean anomaly at time of computation [radians]
        f : True anomaly at time of computation [radians]
    """
 
    # The Parabolic Mean Anomaly is calculated as a function of time
    Mp = calc_Mp(q, t_mjd, tp_mjd)

    # For this type of orbit, we have only to solve a cubic equation for the 
    # tangent of true anomlay halved (see https://en.wikipedia.org/wiki/Parabolic_trajectory)

    # z, an auxiliary quantity is calculated
    z = np.cbrt(1.5*Mp+(np.sqrt(1+(pow(1.5*Mp,2)))))

    # This is the solution of the cubic equation
    tan_fdiv2 = z - (1/z)

    # The true anomaly is calculated 
    f = norm_rad(2*arctan(tan_fdiv2))

    # The angular momentum is calculated just based on q parameter
    h = np.sqrt(mu*2*q)
    h_xyz = np.array([0,0,h])

    # The modulus of the radio vector is calculated applying the
    # polar orbit equation
    r = 2*q/(1 + np.cos(f))

    # Once the polar coordinates are known i the Perifocal frame,
    # the cartesian coordinates in that frame are calculated.
    r_xyz = np.array([r*cos(f), r*sin(f), 0.0])

    # The velocity vector is also calculated applying the same equations
    # as for the other types of orbits.
    rdot_xyz = np.array([-mu*np.sin(f)/h, mu*(1+np.cos(f))/h , 0.0]) 

    return r_xyz, rdot_xyz, r, h_xyz, Mp, f

if __name__ == "__main__" :
    None    