"""This module contains functions related to the calculation of Keplerian orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
"""
# Standard library imports
from math import isclose
import logging
from functools import partial
from math import isclose

# Third party imports
import numpy as np
from numpy import sqrt, cos, sin, cosh, sinh,  arctan, tanh
from scipy.optimize import newton

# Local application imports
from myorbit.util.timeut import  norm_dg
from myorbit.util.constants import *

logger = logging.getLogger(__name__)

# For an elliptic orbit:
#   (Energy <0) 
#   0 < eccentricity  < 1

# For an hyperbolic orbit:
#   (Energy > 0) 
#   eccentricity > 1


def calc_M (t_mjd, tp_mjd, a):
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

    M = np.sqrt(GM/np.power(a,3)) * (t_mjd - tp_mjd)
    return M

def _F(e, M, H):
    """Definition of the Kepler equation. Normally, given a time t, the mean anomaly
    M is calculated and after that the Kepler equation is solved to obtain the Eccentric Anomaly.
    Once we have the Eccentric anomaly, the True anomaly is easily computed.

    Parameters
    ----------
    e : float
        Eccentricity
    M : float
        Mean anomaly
    H : float
        Hyperbolic Anomaly

    Returns
    -------
    float
        The Eccentric anomaly 
    """
    return e*sinh(H)-H-M

def _Fprime(e, H):
    """First derivative with respect to E of the Kepler equation. It is needed for
    solving the Kepler equation more efficently

    Parameters
    ----------
    e : float
        Eccentricity
    E : float
        Eccentric anomaly

    Returns
    -------
    float
        The first derivative 
    """
    return e*cosh(H)-1

def _Fprime2(e, H):
    """Second derivative with respect to E of the Kepler equation. It is needed for
    solving the Kepler equation more efficently

    Parameters
    ----------
    e : float
        Eccentricity
    E : float
        Eccentric anomaly

    Returns
    -------
    float
        The second derivative 
    """
    return e*sinh(H)    

def solve_kepler_eq(e, M, H0):
    """Solve the Kepler equation

    Parameters
    ----------
    e : float
        Eccentricity
    M : float
        Mean anomaly
    E0 : float
        The initial Eccentric anomaly

    Returns
    -------
    Tuple
        A tuple (x,root) where:
            x is The Eccentric anomaly that solves the Kepler equation
            root is a structure with information about how the calculation was, including a flag
            for signaling if the method converged or not.
            In case of the solution does not coverge, the last value obtained is returned

    """
    # The two first parameters of F are bounded, so we end up with f(E)
    # So f(E) = 0 is the equation that is desired to solve (the kepler equation)
    f = partial(_F, e , M)
    # The first parameter of Fprime is bounded, so we end up with fprime(E)
    # According to the newton method, it is better if the first derivative of f is available
    fprime = partial (_Fprime, e)
    # The first parameter of Fprime2 is bounded, so we end up with fprime2(E)
    # According to the newton method, it is better if the second derivative of f is available
    # If the second derivative is provided, the method used for newton method is Halley method
    # is applied that converged in a cubic way
    fprime2= partial (_Fprime2, e)
    x, root = newton(f, H0, fprime, tol=1e-12, maxiter=50, fprime2=fprime2, full_output=True)
    if not root.converged:
       logger.error(f'Not converged with root:{root}') 
    return x, root 

def _calc_H0(e, M):
    """According to Orbital Mechanics (Conway) pg 32, this is a better
    way to provide the initial value for solving Kepler equation in the elliptical
    case.

    Parameters
    ----------
    e : float
        Eccentricity
    M : float
        Mean anomaly (radians)

    Returns
    -------
    float
        The inital value of the Eccentric anomaly to solve the Kepler equation
    """
    return M

def calc_f (H, e):
    tan_fdiv2 = np.sqrt((e+1)/(e-1))*tanh(H/2)
    return 2*arctan(tan_fdiv2)

def calc_rv_for_hyperbolic_orbit (tp_mjd, a_neg, e, t_mjd):
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
    tuple (r_xyz, rdot_xyz, r, h, M, f, H):
        r_xyz: is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane (perifocal frame) [AU]
        rdot_xyz: is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane (perifocal frame) [AU/days]
        r : Modulus of the radio vector of the object (r_xyz) but calculated following the polar equation [AU]    
        h : Angular momentum (deduced from geometic properties)
        M : Mean anomaly at time of computation [radians]
        f : True anomaly at time of computation [radians]
        H : Hyperbolic anomaly [radians]
    """
    cte = sqrt(GM/-a_neg)
    # I have done the calculation and it is right
    # 2*pi/T == cte/a  so this is equivalent to the mean calculation done previously
    # Mean anomaly for the hyperbolic orbit
    Mh = cte*(t_mjd-tp_mjd)/-a_neg 
    
    # Mean anomaly as a function of time (t_mjd) is calculated
    M = calc_M(t_mjd, tp_mjd, np.abs(a_neg))

    # After that we need to solve the Kepler equation to obtain
    # the Hyperbolic Anomaly. For that purpose, we need to provide
    # an initial guess for H, i.e, H0
    H0 =  _calc_H0(e, M) 

    # The Kepler equation is solved so Eccentric Anomaly is obtained
    H, root = solve_kepler_eq(e, M, H0)
    if not root.converged :
        msg = f"Not converged: {root}"
        print (msg)
        logger.error(msg)

    f = calc_f(H,e)   
    r = a_neg*(1-e*e)/(1+e*cos(f))
    r_xyz = np.array([r*cos(f), r*sin(f), 0.0])
    # Semi-latus rectum. Valid for Parabolae, Hyperbolae and Ellipses
    p = a_neg*(1-e*e)
    h = np.sqrt(p*GM)
    rdot_xyz = np.array([-GM*np.sin(f)/h,GM*(e+np.cos(f))/h , 0.0]) 

    # Alternative computation
    cosh_H = cosh(H)
    sinh_H = sinh(H)

    fac =  sqrt((e+1.0)*(e-1.0))
    rho = e*cosh_H - 1.0
    r1 = a_neg*(1-e*cosh_H)    
    r1_xyz = np.array([np.abs(a_neg)*(e-cosh_H), np.abs(a_neg)*fac*sinh_H, 0.0])
    r1dot_xyz = np.array([-cte*sinh_H/rho, cte*fac*cosh_H/rho,0.0])

    if not np.allclose(r_xyz, r1_xyz, atol=1e-012):
        print (f'Differences between r1: {r} and r2:{r1}')   

    if not np.allclose(rdot_xyz, r1dot_xyz, atol=1e-012):
        print (f'Differences between r1: {rdot_xyz} and r2:{r1dot_xyz}')

    return r_xyz, rdot_xyz, r, h, M, f, H

if __name__ == "__main__" :
    None
    