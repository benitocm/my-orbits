"""This module contains functions related to the calculation of hyperbolic orbits
It is based on the books 
     "Orbital Mechanichs for engineering students"
     "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
and also in 
    "Spacecraft Dynamics and Control" lectures of Mattew M. Peet
"""
# Standard library imports
from math import isclose
import logging
from functools import partial
from math import isclose
from typing import MutableSequence

# Third party imports
import numpy as np
from numpy import sqrt, cos, sin, cosh, sinh,  arctan, tanh
from scipy.optimize import newton

# Local application imports
from myorbit.util.general import NoConvergenceError, mu_Sun
from myorbit.util.timeut import norm_rad


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

#
#  True Anomaly is still the angle the position vector r^ makes with the
#  eccentricity vector, e^, measured counter-clockwise
#
# The orbit does not repeat (no period, T ), we can't use it. 
# No reference circle so Eccentric Anomly is undefined
#
# Hyperbolic anomaly (H) is the hyperbolic angle using the area enclosed by the
# center of the hyperbola, the point of perifocus and the point on the reference
# hyperbola directly above the position vector

def calc_M (t_mjd, tp_mjd, a, mu=mu_Sun):
    """Computes the mean anomaly as a function of time (t_mjd).

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

    #M = np.sqrt(GM/np.power(a,3)) * (t_mjd - tp_mjd)

    M = sqrt(mu/a)*(t_mjd - tp_mjd)/a
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
        Mean anomaly [radians]
    H : float
        Hyperbolic Anomaly  [radians]

    Returns
    -------
    float
        Hyperbolic Anomaly 
    """
    return e*sinh(H)-H-M

def _Fprime(e, H):
    """First derivative with respect to H of the Kepler equation. It is needed for
    solving the Kepler equation more efficently

    Parameters
    ----------
    e : float
        Eccentricity
    H : float
        Hyperbolic Anomaly  [radians]

    Returns
    -------
    float
        The first derivative 
    """
    return e*cosh(H)-1

def _Fprime2(e, H):
    """Second derivative with respect to H of the Kepler equation. It is needed for
    solving the Kepler equation more efficently

    Parameters
    ----------
    e : float
        Eccentricity
    E : float
        Hyperbolic Anomaly  [radians]

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
        Mean anomaly [radians]
    H0 : float
        The initial Hyperbolic anomaly [radians]

    Returns
    -------
    Tuple
        A tuple (x,root) where:
            x is The Eccentric anomaly that solves the Kepler equation
            root is a structure with information about how the calculation was, including a flag
            for signaling if the method converged or not.
            In case of the solution does not coverge, the last value obtained is returned

    """
    # The two first parameters of F are bounded, so we end up with f(H)
    # So f(H) = 0 is the equation that is desired to solve (the kepler equation)
    f = partial(_F, e , M)
    # The first parameter of Fprime is bounded, so we end up with fprime(H)
    # According to the newton method, it is better if the first derivative of f is available
    fprime = partial (_Fprime, e)
    # The first parameter of Fprime2 is bounded, so we end up with fprime2(H)
    # According to the newton method, it is better if the second derivative of f is available
    # If the second derivative is provided, the method used for newton method is Halley method
    # is applied that converged in a cubic way
    fprime2= partial (_Fprime2, e)
    x, root = newton(f, H0, fprime, tol=1e-12, maxiter=50, fprime2=fprime2, full_output=True)
    if not root.converged:        
       logger.error(f'Hiperbolical Kepler equation not converged with root:{root}') 
       raise NoConvergenceError(x, root.function_calls, root.iterations)
    # Checking the solution 
    if not isclose(f(x), 0.0, rel_tol=0, abs_tol=1e-08):
        msg = f'Hiperbolical Kepler equation not solution found with Laguerre with root: {x} and error: {abs_tol}'
        logger.error(msg) 
        raise NoConvergenceError(x, root.iterations, root.iterations, H0)
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
        Mean anomaly [radians]

    Returns
    -------
    float
        The inital value of the Hyperbolic anomaly to solve the Kepler equation
    """
    if M>=0 :
        return np.abs(np.log(1.8+(2*M/e)))
    else :
        return -np.log(1.8-(2*M/e))

def calc_f (H, e):
    """Computes the True anomaly given the Hyperbolic anomaly.
    The true anomaly must be between [0,2*PI)

    Parameters
    ----------
    H : float
        Hyperbolic anomaly [radians]
    e : float
        Eccentricity

    Returns
    -------
    float
        The true anomaly [radians]
    """
    tan_fdiv2 = np.sqrt((e+1)/(e-1))*tanh(H/2)
    return norm_rad(2*arctan(tan_fdiv2))

def calc_rv_for_hyperbolic_orbit (tp_mjd, a_neg, e, t_mjd, mu=mu_Sun):
    """Computes the state vector and other quantities. The time evolution comes from M (Mean anomaly).
    The computation of the mean anomaly is outside of this method because it depends on the type of object (asteroids or
    comets)

    Parameters
    ----------
    M : float 
        Mean anomaly at computation time [radians]
    a : float
        Semimajor axis of the orbit [AU]
    e : float
        Eccentricity

    Returns
    -------
    tuple (r_xyz, rdot_xyz, r, h, M, f, H):
        r_xyz: is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane (perifocal frame) [AU]
        rdot_xyz: is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane (perifocal frame) [AU/days]
        r : Modulus of the radio vector of the object (r_xyz) but calculated following the polar equation [AU]    
        h_xyz : Angular momentum (deduced from geometic properties)
        M : Mean anomaly at time of computation [radians]
        f : True anomaly at time of computation [radians]
        H : Hyperbolic anomaly at time of computation [radians]
    """
    cte = sqrt(mu/-a_neg)
    # I have done the calculation and it is right
    # 2*pi/T == cte/a  so this is equivalent to the mean calculation done previously
    # Mean anomaly for the hyperbolic orbit
    #Mh = cte*(t_mjd-tp_mjd)/-a_neg 
    
    # Mean anomaly as a function of time (t_mjd) is calculated
    Mh = calc_M(t_mjd, tp_mjd, np.abs(a_neg))

    # After that we need to solve the Kepler equation to obtain
    # the Hyperbolic Anomaly. For that purpose, we need to provide
    # an initial guess for H, i.e, H0
    H0 =  _calc_H0(e, Mh) 

    # The Kepler equation is solved so Eccentric Anomaly is obtained
    H, _ = solve_kepler_eq(e, Mh, H0)

    # From H, we obtain the True Anomaly as f
    f = calc_f(H,e)   
    # The polar coordinates of the body w.r.t the Perifocal Frame are r and f
    # We move from polar to cartesian inside the Perifocal Frame, the z coordinate is 0 because
    # we are in a plane# 
    # The polar equation of the orbit is used to calculate the modulus of r^
    r = a_neg*(1-e*e)/(1+e*cos(f))
    # The polar coordinates of the body w.r.t the Perifocal Frame are r and f
    # We move from polar to cartesian inside the Perifocal Frame, the z coordinate is 0 because
    # we are in a plane# 
    r_xyz = np.array([r*cos(f), r*sin(f), 0.0])
    # Semi-latus rectum. Valid for Parabolae, Hyperbolae and Ellipses.
    # is computed to calculte the angular momentum 
    p = a_neg*(1-e*e)    
    # The modulus of angular momentum is computed from from geometric data
    h = np.sqrt(p*mu)
    h_xyz = np.array([0,0,h])

    # To calculate the velocity (in cartesian coordinates) we use the formula in Orbital Mechanics for 
    # Students (eq. 2.123 y 2.124) based on the modulus of angular momentum and true anomaly
    rdot_xyz = np.array([-mu*np.sin(f)/h,mu*(e+np.cos(f))/h , 0.0]) 
    
    # Alternative computation without calculating directly the true anomaly 
    # described in the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger (pag. 80)

    cosh_H = cosh(H)
    sinh_H = sinh(H)

    fac =  sqrt((e+1.0)*(e-1.0))
    rho = e*cosh_H - 1.0
    r1 = a_neg*(1-e*cosh_H)    
    r1_xyz = np.array([np.abs(a_neg)*(e-cosh_H), np.abs(a_neg)*fac*sinh_H, 0.0])
    r1dot_xyz = np.array([-cte*sinh_H/rho, cte*fac*cosh_H/rho, 0.0])

    if not np.allclose(r_xyz, r1_xyz, atol=1e-012):
        logger.warning (f'Difference in r_xyz between the two alternatives ={np.linalg.norm(r_xyz- r1_xyz)}')   
    if not np.allclose(rdot_xyz, r1dot_xyz, atol=1e-012):
        logger.warning (f'Difference in rdot_xyz between the two alternatives ={np.linalg.norm(rdot_xyz- r1dot_xyz)}')   

    return r_xyz, rdot_xyz, r, h_xyz, Mh, f, H

if __name__ == "__main__" :
    None
    