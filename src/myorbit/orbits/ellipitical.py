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
from numpy import sqrt, cos, sin
from scipy.optimize import newton

# Local application imports
from myorbit.util.timeut import norm_dg, norm_rad
from myorbit.util.general import pow
from myorbit import coord as co

np.set_printoptions(precision=12)

from myorbit.util.constants import *

logger = logging.getLogger(__name__)

# For an elliptic orbit:
#   (Energy <0) 
#   0 < eccentricity  < 1

# For an hyperbolic orbit:
#   (Energy > 0) 
#   eccentricity > 1

def calc_M_for_body(t_mjd, epoch_mjd, a, M_at_epoch) :
    
    """Computes the mean anomaly based on the data of BodyElms, in this case,
    uses the period (calculated) and the Mean anomaly at epoch.

    Parameters
    ----------
    t_mjd : float
        Time of the computation as Modified Julian Day
    epoch_mjd : float
        The Epoch considered
    period_in_days : float
         period of the orbit [days]
    M_at_epoch : float
        The mean anomaly at epoch [radians]

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
        Mean anomaly
    E : float
        Eccentric anomaly

    Returns
    -------
    float
        The Eccentric anomaly 
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
        Eccentric anomaly

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
        Eccentric anomaly

    Returns
    -------
    float
        The second derivative 
    """
    return e*sin(E)    

def solve_kepler_eq(e, M, E0):
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
        Mean anomaly (radians)

    Returns
    -------
    float
        The inital value of the Eccentric anomaly to solve the Kepler equation
    """
    mu = M+e
    num = M*(1 - np.sin(mu) ) + mu*np.sin(M)
    den = 1 + np.sin(M) - np.sin(mu)
    return num/den


def calc_rv_for_elliptic_orbit (M, a, e):
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

    # Mean anomaly as a function of time (t_mjd) is calculated
    #M = calc_M(t_mjd, tp_mjd, a)

    # After that we need to solve the Kepler equation to obtain
    # the Eccentric Anomaly. For that purpose, we need to provide
    # an initial guess for E, i.e, E0
    E0 =  _calc_E0(e, M) 

    # The Kepler equation is solved so Eccentric Anomaly is obtained
    E, root = solve_kepler_eq(e, M, E0)
    if not root.converged :
        msg = f"Not converged: {root}"
        print (msg)
        logger.error(msg)

    # From E, we obtain the True Anomaly as f
    cos_f = (cos(E) - e)/(1 - e*cos(E))
    f = np.arccos(cos_f)
    # Because arccos returns an angle between [0,PI] and True Anomaly goes
    # between [0,2*PI) we need to solve the quadrant ambiguity. Because
    # Mean Anomaly and True Anomaly are in the same quadrant we can use 
    # that fact to resolve it.
    if PI <= M < TWOPI:
        f = TWOPI - f

    sin_f = np.sin(f) 
    
    # Once we have f, we can calculate the modulus of the position vector r
    # using the orbit polar equation 
    r = a*(1-e*e)/(1+e*cos_f)

    # At this point, we have the polar coordinates of the body w.r.t the
    # Perifocal Frame

    # We move from Polar to cartesian in the Perifocal Frame, the z coordinate is 0 because
    # we are in the Perifocal Frame
    r_xyz = np.array([r*cos_f, r*sin_f, 0.0])  

    # To calculate the velocity (in cartesian coordinates) we use the formula in Orbital Mechanics for 
    # Students (eq. 2.123 y 2.124). For that, we need first the 
    # Angular Momentum
    h = np.sqrt(GM*a*(1-e*e))

    rdot_xyz = np.array([-GM*sin_f/h,GM*(e+cos_f)/h , 0.0]) 

    # Double checking:
    #   The Angular Momentum h should be constant during the orbit
    #   The modulus of rdot_xyz should be the same to the one calculated from the Energy velocty:
    #        - Energy = - GM/(2*a)
    #        - v = sqrt(2*(Energy+(GM/r)))
    #   The Angular momentum h should be the sames r x v   {np.cross(r_xyz,rdot_xyz)}")
    #)

    return r_xyz, rdot_xyz, r, h, M, f, E

def test_comet():
    # For comets, we have Perihelion distance (q or rp) instead of semimajor axis (a)
    # For asteroids, we have the semimajor axis (a)    t_mjd = 56197.0
    e = 0.99999074
    TP_MJD = 56198.22249000007
    T0_MJD = 56197.0
    q = 1.29609218 # Perihelion distance
    a = None
    if a is None :
        a = q / (1-e) 
    hs= []
    for dt in range(0,10):
        t_mjd = T0_MJD + dt
        r_xyz, rdot_xyz, M, f, E, h_geo, r = calc_rv_for_elliptic_orbit(TP_MJD, a, e)
        #print (r_xyz, rdot_xyz, np.rad2deg(M), np.rad2deg(f), np.rad2deg(E)) 
        h_rv = np.cross(r_xyz,rdot_xyz)
        #print (h_geo, h_rv[2], isclose(h_geo, h_rv[2], abs_tol=1e-12))
        hs.append(h_rv[2])
        print (np.rad2deg(f),np.rad2deg(M))
        print (quadrant(f), quadrant(M))
        Energy = - GM/(2*a)
        v = sqrt(2*(Energy+(GM/r)))
        #print (v, np.linalg.norm(rdot_xyz), isclose(v,np.linalg.norm(rdot_xyz),abs_tol=1e-12))
    is_h_cte = all(isclose(h, hs[0], abs_tol=1e-12) for h in hs)
    print (is_h_cte)

def test_body():
     None   


    




if __name__ == "__main__" :
    test_comet()
    