"""This module contains functions related to the calculation of elliptical orbits
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

# Third party imports
import numpy as np
from numpy import sqrt, cos, sin
from scipy.optimize import newton

# Local application imports
from myorbit.util.timeut import  norm_rad
from myorbit.util.general import pow
from myorbit.util.constants import *

logger = logging.getLogger(__name__)

# For an elliptic orbit:
#   (Energy <0) 
#   0 < eccentricity  < 1

# For an hyperbolic orbit:
#   (Energy > 0) 
#   eccentricity > 1

def calc_tp(M_at_epoch, a, epoch):
    """Compute the perihelion passage given Mean anomaly and the epoch

    Parameters
    ----------
    M_at_epoch : [type]
        [description]
    a : [type]
        [description]
    epoch : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Time taken to go from the Mean anomaly at epoch up to the perihelion
    deltaT = TWOPI*np.sqrt(pow(a,3)/GM)*(1-M_at_epoch/TWOPI)    
    return deltaT + epoch

def calc_M_for_body(t_mjd, epoch_mjd, a, M_at_epoch) :
    
    """Computes the mean anomaly as a function of time (t_mjd). This method is used
    when the Time of perihelion passage is unknown (for asteroids). 

    Parameters
    ----------
    t_mjd : float
        Time of the computation [Modified Julian day]
    epoch_mjd : float
        Epoch of the orbital elemnts [Modified Julian day]
    a : float
        Semimajor axis of the orbit [AU]
    M_at_epoch : float
        The mean anomaly at the time of epoch (epoch_mjd) [radians]

    Returns
    -------
    float
        The mean anomaly [radians]
    """
    period_in_days = TWOPI*sqrt(pow(a,3)/GM)
    M = (t_mjd - epoch_mjd)*TWOPI/period_in_days
    M += M_at_epoch
    return norm_rad(M)

def calc_M (t_mjd, tp_mjd, a):
    """Computes the mean anomaly as a function of time (t_mjd). This method is used
    when the Time of perihelion passage is known (for comets)

    Parameters
    ----------
    t_mjd : float
        Time of the computation [Modified Julian day]
    tp_mjd : float
        Time of perihelion passage [Modified Julian day]
    a : float
        Semimajor axis [AU]
    
    Returns
    -------
    float
        The mean anomaly [radians]
    """    

    M = np.sqrt(GM/pow(a,3)) * (t_mjd - tp_mjd)
    return norm_rad(M)

def _F(e, M, E):
    """Definition of the Kepler equation. Normally, given a time t, the mean anomaly
    M is calculated and after that the Kepler equation is solved to obtain the Eccentric Anomaly.
    Once we have the Eccentric anomaly, the True anomaly is easily computed.

    Parameters
    ----------
    e : float
        Eccentricity
    M : float
        Mean anomaly [radians]
    E : float
        Eccentric anomaly [radians]

    Returns
    -------
    float
        The Eccentric anomaly [radians]
    """
    return E-e*sin(E)- M

def _Fprime(e, E):
    """First derivative with respect to E of the Kepler equation. It is needed for
    solving the Kepler equation more efficently

    Parameters
    ----------
    e : float
        Eccentricity
    E : float
        Eccentric anomaly [radians]

    Returns
    -------
    float
        The first derivative 
    """
    return 1 - e*cos(E)

def _Fprime2(e, E):
    """Second derivative with respect to E of the Kepler equation. It is needed for
    solving the Kepler equation more efficently

    Parameters
    ----------
    e : float
        Eccentricity
    E : float
        Eccentric anomaly [radians]

    Returns
    -------
    float
        The second derivative 
    """
    return e*sin(E)    

def solve_kepler_eq(e, M, E0):
    """Solve the Kepler equation numerically

    Parameters
    ----------
    e : float
        Eccentricity
    M : float
        Mean anomaly [radians]
    E0 : float
        The initial Eccentric anomaly [radians]

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
    x, root = newton(f, E0, fprime, tol=1e-12, maxiter=50, fprime2=fprime2, full_output=True)
    if not root.converged:
       logger.error(f'Not converged with root:{root}') 
    return x, root 

def _calc_E0(e, M):
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
        The inital value of the Eccentric anomaly to solve the Kepler equation [radians]
    """
    mu = M+e
    num = M*(1 - np.sin(mu) ) + mu*np.sin(M)
    den = 1 + np.sin(M) - np.sin(mu)
    return num/den

def calc_rv_for_elliptic_orbit (M, a, e):
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
    tuple (r_xyz, rdot_xyz, r, h, M, f, E):
        r_xyz: is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane (perifocal frame) [AU]
        rdot_xyz: is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane (perifocal frame) [AU/days]
        r : Modulus of the radio vector of the object (r_xyz) but calculated following the polar equation [AU]    
        h : Angular momentum (deduced from geometic properties)
        M : Mean anomaly at time of computation [radians]
        f : True anomaly at time of computation [radians]
        E : Eccentric anomaly at time of computation [radians]
    """
    # Once the mean anomaly is known (depends of the time of computation), it is
    # needed to solve the Kepler equation to obtain the Eccentric Anomaly. For that purpose,
    # we need to provide an initial guess for E, i.e, E0
    E0 =  _calc_E0(e, M) 

    # The Kepler equation is solved so Eccentric Anomaly to obtain Eccentric Anomaly
    E, root = solve_kepler_eq(e, M, E0)
    if not root.converged :
        msg = f"Not converged: {root}"
        print (msg)
        logger.error(msg)

    # From E, we obtain the True Anomaly as f
    cos_f = (cos(E) - e)/(1 - e*cos(E))
    f = np.arccos(cos_f)
    # Because arccos returns an angle between [0,PI] and True Anomaly goes
    # between [0,2*PI) we need to solve the hemisphere ambiguity. Because
    # Mean Anomaly and True Anomaly are in the same hemisphere we can use 
    # that fact to resolve it.
    if PI <= M < TWOPI:
        f = TWOPI - f

    sin_f = np.sin(f) 
    
    # Once f (true anomaly) has been calculated, the modulus of the position vector r
    # can be calculated using the orbit polar equation (valid for the three conics)
    r = a*(1-e*e)/(1+e*cos_f)

    # The polar coordinates of the body w.r.t the Perifocal Frame are r and f
    # We move from polar to cartesian inside the Perifocal Frame, the z coordinate is 0 because
    # we are in a plane
    r_xyz = np.array([r*cos_f, r*sin_f, 0.0])  

    # To calculate the velocity (in cartesian coordinates) we use the formula in Orbital Mechanics for 
    # Students (eq. 2.123 y 2.124). For that, we need first to compute the Angular Momentum using
    # geometric properties
    h = np.sqrt(GM*a*(1-e*e))

    rdot_xyz = np.array([-GM*sin_f/h,GM*(e+cos_f)/h , 0.0]) 

    # Double checking:
    #   The Angular Momentum h should be constant during the orbit
    #   For each calculation, the modulus of rdot_xyz should be the same to the 
    #   one calculated from the Energy velocity:
    #        - Energy = - GM/(2*a)
    #        - v = sqrt(2*(Energy+(GM/r)))
    #   The Angular momentum h should be the sames r x v   {np.cross(r_xyz,rdot_xyz)}")

    return r_xyz, rdot_xyz, r, h, M, f, E


if __name__ == "__main__" :
    None
    