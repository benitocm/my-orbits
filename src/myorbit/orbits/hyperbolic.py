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

    if not np.allclose(r_xyz,r1_xyz,atol=1e-012):
        print (f'Differences between r1: {r} and r2:{r1}')   

    if not np.allclose(rdot_xyz,r1dot_xyz,atol=1e-012):
        print (f'Differences between r1: {rdot_xyz} and r2:{r1dot_xyz}')

    return r_xyz, rdot_xyz, r, h, M, f, H


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

def _parabolic_orbit (tp, q, e, t, max_iters=15):
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
    tuple (r,v):
        Where r is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane [AU]
        Where v is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane [AU/days]
    """
    E_2 = 0.0    
    factor = 0.5 * e
    cte = sqrt(GM/(q*(1.0+e)))
    tau = sqrt(GM)*(t-tp)
    epsilon = 1e-7

    for i in range(max_iters):
        E20 = E_2 
        A = 1.5 * sqrt(factor/(q*q*q))*tau
        B = np.cbrt(sqrt(A*A+1.0)+A)
        u = B - 1.0/B 
        u_2 = u*u
        E_2 = u_2*(1.0-e)/factor 
        c1, c2, c3 = _calc_stumpff_values(E_2)
        factor = 3.0*e*c3 
        if isclose(E_2, E20, abs_tol=epsilon) :
            R = q * (1.0 + u_2*c2*e/factor)
            r = np.array([q*(1.0-u_2*c2/factor), q*sqrt((1.0+e)/factor)*u*c1,0.0])
            v = np.array([-cte*r[1]/R, cte*(r[0]/R+e),0.0])
            return r,v,1,1,1
    #logger.warning(f"Not converged after {i} iterations")
    logger.error(f'Not converged with q:{q},  e:{e}, t:{t}, t0:{tp} after {i} iterations')
    return 0,0,0,0,0


def test_comet():
    # For comets, we have Perihelion distance (q or rp) instead of semimajor axis (a)
    # For asteroids, we have the semimajor axis (a)   
    e = 1.06388423
    TP_MJD = 59311.54326000018 
    T0_MJD = 59205.0
    q = 3.20746664 # Perihelion distance
    a = None
    if a is None :
        a = q / (1-e) 
    hs= []
    for dt in range(0,500):
        t_mjd = T0_MJD + dt
        r_xyz, rdot_xyz, M, f, H, r1, r2,h_geo  = calc_rv_for_hyperbolic_orbit(t_mjd, TP_MJD, a, e)
        print (f't:{t_mjd}, M:{np.rad2deg(M)}, f:{np.rad2deg(f)}, H:{np.rad2deg(H)}, r1:{r1}, r2:{r2}') 
        h_rv = np.cross(r_xyz,rdot_xyz)
        print (h_geo, h_rv[2], isclose(h_geo, h_rv[2], abs_tol=1e-12))
        hs.append(h_rv[2])
        #print (np.rad2deg(f),np.rad2deg(M))
        #print (quadrant(f), quadrant(M))
        Energy = - GM/(2*a)
        v = sqrt(2*(Energy+(GM/r1)))
        #print (v, np.linalg.norm(rdot_xyz), isclose(v,np.linalg.norm(rdot_xyz),abs_tol=1e-12))
    is_h_cte = all(isclose(h, hs[0], abs_tol=1e-12) for h in hs)
    print (is_h_cte)

def test_body():
     None   


    




if __name__ == "__main__" :
    test_comet()
    