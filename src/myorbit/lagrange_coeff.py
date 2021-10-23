"""
This module contains functions related to Lagrange Coefficients and universal variables
The source comes from the book Orbital Mechanics for Engineering Students 
"""
# Standard library imports

from functools import partial
from math import isclose
import logging
import sys

# Third party imports
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, arctan, sqrt,cosh,sinh,deg2rad,rad2deg, sqrt
from numpy.linalg import multi_dot, norm
from toolz import pipe, compose, first, valmap

# Local application imports
import myorbit.orbits.orbutil as ob
import myorbit.data_catalog as dc
from myorbit.orbits.ephemeris_input import EphemrisInput
from myorbit.util.general import pow
import myorbit.util as ut
from myorbit.util.constants import *
import  myorbit.kepler_u as ku

logger = logging.getLogger(__name__)


def stump_C(z) :
    """
    Evaluates the Stumpff function C(z) according to the Equation 3.53
    
    Args:
        z : argument (real number)

    Returns :
        The value of the C(z)
    """

    if z > 0 :
        return (1 - cos(sqrt(z)))/z        
    elif z < 0 :
        return (cosh(sqrt(-z)) - 1)/(-z)
    else :
        return 0.5


def stump_S(z) :
    """
    Evaluates the Stumpff function S(z) according to the Equation 3.52
    
    Args:
        z : argument (real number)

    Returns :
        The value of the S(z)
    """

    if z > 0:
        sz = sqrt(z) 
        return (sz - sin(sz))/pow(sz,3)
    elif z < 0 :
        s_z = sqrt(-z) 
        # According to the equation the denominatori is pow(sqrt(z),3)
        return  (sinh(s_z) - s_z)/pow(s_z,3)
    else :
        return 0.1666666666666666


def kepler_U_prv(mu, x , dt, ro, vro, inv_a, nMax=500):
    """
    Compute the general anomaly by solving the universal Kepler
    function  using the Newton's method 

    Args:
        mu : Gravitational parameter (AU^3/days^2)
        dt : time since x = 0 (days)
        ro : radial position (AU) when x=0
        vro: rdial velocity (AU/days) when x=0
        inv_a : reciprocal of the semimajor axis (1/AU)
        nMax : maximum allowable number of iterations

    Returns :
        The universal anomaly (x) AU^.5
    """

    error = 1.0e-8
    n = 0
    ratio = 1
    while (abs(ratio) > error) and  (n <= nMax) :
        n = n + 1
        z = x*x
        C = stump_C(inv_a*z)
        S = stump_S(inv_a*z)
        F = ro*vro/sqrt(mu)*z*C + (1 - inv_a*ro)*z*x*S + ro*x - sqrt(mu)*dt
        dFdx = ro*vro/sqrt(mu)*x*(1 - inv_a*z*S) + (1 - inv_a*ro)*z*C + ro
        ratio = F/dFdx
        x = x - ratio
    if n > nMax :
        return (False,x,ratio)
    else :
        return (True,x,ratio)

LINEAR_GRID = list(np.linspace(2.5,4,16,endpoint=True))

def kepler_U(mu, dt, ro, vro, inv_a, nMax=500):
    """
    Compute the general anomaly by solving the universal Kepler
    function  using the Newton's method 

    Args:
        mu : Gravitational parameter (AU^3/days^2)
        dt : time since x = 0 (days)
        ro : radial position (AU) when x=0
        vro: rdial velocity (AU/days) when x=0
        inv_a : reciprocal of the semimajor axis (1/AU)
        nMax : maximum allowable number of iterations

    Returns :
        The universal anomaly (x) AU^.5
    """

    """
    ratios = []
    # For some parabolic comets, using some initial values improves the convergence
    for x in [sqrt(mu)*abs(inv_a)*dt]: #+ LINEAR_GRID :
        converged, result, ratio = kepler_U_prv(mu, x , dt, ro, vro, inv_a, nMax=1000)
        if converged:
            return result 
        else :
            ratios.append(str(ratio))
    logger.error(f"Number max iteration reached but not converged, ratios: {','.join(ratios)}")
    return result 
    """
    x = sqrt(mu)*abs(inv_a)*dt
    return ku.kepler_U(mu, x, dt, ro, vro, inv_a, nMax)
   


def calc_f_g(mu, x, t, ro, inv_a):
    """
    Calculates the Lagrange f and g coefficients starting from the initial position r0 (radio vector
    from the dinamical center (normally the Sun) and the elapsed time t)

    Args:
        mu : Gravitational parameter (AU^3/days^2)
        x : the universal anomaly after time t (km^0.5)
        t : the time elapsed since ro (days)
        ro : the radial position at time to (AU)
        inv_a : reciprocal of the semimajor axis (1/AU)

    Returns :
        A tuple with f and g coefficients
    """
    z = inv_a*pow(x,2)
    f = 1 - pow(x,2)/ro*stump_C(z)    
    g = t - 1/sqrt(mu)*pow(x,3)*stump_S(z)
    return f, g 

def calc_fdot_gdot(mu, x, r, ro, inv_a) :
    """
    Calculates the time derivatives of Lagrange coefficients
    f and g coefficients.

    Args:
        mu : Gravitational parameter (AU^3/days^2)
        x : the universal anomaly after time t (AU^0.5)
        r :  the radial position (radio vector) after time t (AU)
        ro : the radial position (radio vector) at time to (AU)
        inv_a : reciprocal of the semimajor axis (1/AU)

    Returns :
        a tuple with fdot and gdot 
    """

    z = inv_a*pow(x,2)
    #%...Equation 3.69c:
    fdot = sqrt(mu)/r/ro*(z*stump_S(z) - 1)*x
    # %...Equation 3.69d:
    gdot = 1 - pow(x,2)/r*stump_C(z)
    return fdot, gdot


def rv_from_r0v0(mu, R0, V0, t):
    """
    This function computes the state vector (R,V) from the
    initial state vector (R0,V0) and after the elapsed time.
    Internally uses the universal variables and the lagrange coefficients.
    Although according to the book, this is used in the perifocal plane (i.e. the orbital plane),
    in the enckle method I used in the ecliptic plane and it works. It may be becasue at the
    end the the size of the orbital plane does not change only it is rotated according to the
    Gauss angles.

    Args:
        mu : Gravitational parameter (AU^3/days^2)
        R0 : initial position vector (AU)
        V0 : initial velocity vector (AU/days)
         t : Elapsed time (days)

    Returns :
        A tuple with:
            R: Final position vector (AU)
            V: Final position vector (AU/days)
    """
    #...Magnitudes of R0 and V0:
    r0 = norm(R0)
    v0 = norm(V0)
    #...Initial radial velocity:
    vr0 = np.dot(R0, V0)/r0
    #...Reciprocal of the semimajor axis (from the energy equation):
    alpha = 2/r0 - pow(v0,2)/mu
    #...Compute the universal anomaly:
    x = kepler_U(mu, t, r0, vr0, alpha)
    #...Compute the f and g functions:
    f, g = calc_f_g(mu, x, t, r0, alpha)
    #...Compute the final position vector:
    R = f*R0 + g*V0
    #...Compute the magnitude of R:
    r = norm(R)
    #...Compute the derivatives of f and g:
    fdot, gdot = calc_fdot_gdot(mu, x, r, r0, alpha)
    #...Compute the final velocity:
    V = fdot*R0 + gdot*V0
    return R, V



def test1() :
    mu = 398600
    R0 = np.array([7000, -12124, 0])
    V0 = np.array([2.6679, 4.6210, 0])
    t = 3600
    R,V = rv_from_r0v0(mu,R0, V0, t)
    print ("R: ",R)
    print ("V: ",V)


def test3():
    mu_sun__m3_s_2 = 1.32712440018e20
    AU_m = 149597870700
    seconds_in_day = 3600*24
    mu_sun_AU3_days = mu_sun__m3_s_2 * seconds_in_day*seconds_in_day/(AU_m*AU_m*AU_m)
    print (mu_sun_AU3_days)


"""
def test2():
    #mu = (1 + 0.000000166)
    #
    #mu = mu_by_name["Sun"] + mu_by_name["Mercury"]
    #mu_sun = ut.GM
    #mu_mercury = 

    R0 = np.array([0.1693419, -0.3559908, -0.2077172])
    V0 = np.array([1.1837591, 0.6697770, 0.2349312]) * ut.k
    
    t0 = 6280.5
    k = 0.017202098895
    for t in range (0,100,10) :
        #r,v = rv_from_r0v0(mu, R0, V0, t*k)
        #r,v = rv_from_r0v0(mu, R0, V0, t)
        print (f"{t0+t}  {r}   {norm(r)} ")
"""
    

if __name__ == "__main__":
    test1()   
    #test4()
    #test3()
    #print (mu_Sun)
    #print (mu_Mercury)
    #print (ut.k_2)


 