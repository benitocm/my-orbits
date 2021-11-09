""" This module contains functions related to Lagrange Coefficients and universal variables
The source comes from the book 'Orbital Mechanics for Engineering Students'
"""
# Standard library imports
from functools import partial
import logging

# Third party imports
import numpy as np
from numpy import sin, cos, sqrt,cosh,sinh, sqrt
from numpy.linalg import norm
from scipy.optimize import newton
from math import isclose
from myorbit.util.constants import TWOPI

# Local application imports
from myorbit.util.general import pow, NoConvergenceError
import  myorbit.lagrange.kepler_u as ku
from myorbit.util.timeut import norm_rad    

from pathlib import Path
CONFIG_INI=Path(__file__).resolve().parents[3].joinpath('conf','config.ini')
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read(CONFIG_INI)
LAGUERRE_ABS_TOL = float(cfg.get('general','laguerre_abs_tol'))

logger = logging.getLogger(__name__)

def stump_C(z) :
    """Evaluates the Stumpff function C(z) according to the Equation 3.53

    Parameters
    ----------
    z : float
        The argument

    Returns
    -------
    float
        The value of the C(z)
    """

    if z > 0 :
        return (1 - cos(sqrt(z)))/z        
    elif z < 0 :
        return (cosh(sqrt(-z)) - 1)/(-z)
    else :
        return 0.5

def stump_S(z) :    
    """Evaluates the Stumpff function S(z) according to the Equation 3.52

    Parameters
    ----------
    z : float
        The argument

    Returns
    -------
    float
        The value of the S(z)
    """
    if z > 0:
        sz = sqrt(z) 
        return (sz - sin(sz))/pow(sz,3)
    elif z < 0 :
        s_z = sqrt(-z) 
        # According to the equation the denominator is pow(sqrt(z),3)
        return  (sinh(s_z) - s_z)/pow(s_z,3)
    else :
        return 0.1666666666666666

def _F(mu, r0, vr0, inv_a, dt, x):
    z = x*x
    C = stump_C(inv_a*z)
    S = stump_S(inv_a*z)
    return  (r0*vr0/sqrt(mu))*z*C + (1 - inv_a*r0)*z*x*S + r0*x - sqrt(mu)*dt

def _Fprime(mu, r0, vr0, inv_a, x):
    z = x*x
    C = stump_C(inv_a*z)
    S = stump_S(inv_a*z)
    return (r0*vr0/sqrt(mu))*x*(1 - inv_a*z*S) + (1 - inv_a*r0)*z*C + r0


def _Fprime2(mu, r0, vr0, inv_a, x):
    z = x*x
    C = stump_C(inv_a*z)
    S = stump_S(inv_a*z)
    return r0*vr0/sqrt(mu) - (inv_a*r0*vr0/sqrt(mu))*pow(x,2)*C + (1-inv_a*r0)*x-(1-inv_a*r0)*inv_a*pow(x,3)*S

    

def solve_kepler_eq(X0, mu, r0, vr0, inv_a, dt):
    # The first 5 parameters of F are bounded, so we end up with f(x)
    # So f(x) = 0 is the equation that is desired to solve (the kepler equation)
    # for universal anomaly
    f = partial(_F, mu, r0, vr0, inv_a, dt)

    # The first 4 parameter of Fprime are bounded, so we end up with fprime(x)
    # According to the newton method, it is better if the first derivative of f is available
    fprime = partial (_Fprime, mu, r0, vr0, inv_a)

    # The first 4 parameter of Fprime2 are bounded, so we end up with fprime2(x)
    # According to the newton method, it is better if the second derivative of f is available
    fprime2 = partial (_Fprime2, mu, r0, vr0, inv_a)

    x, root = newton(f, X0, fprime=fprime, fprime2=fprime2, tol=1e-09, maxiter=700,  full_output=True, disp=False)
    if not root.converged:        
       logger.error(f'Universal Kepler equation not converged with root:{root}') 
       raise NoConvergenceError(x, root.function_calls, root.iterations, X0)
    logger.info(f'Converged in {root.iterations} iterations and {root.function_calls} function_calls for X0={X0}, ro={r0}, vro={vr0}, inv_a={inv_a}, dt={dt}  Not converged with root:{root}') 
    return x, root 

def solve_kepler_universal_laguerre (mu, dt, r0, vr0, inv_a, abs_tol=LAGUERRE_ABS_TOL, max_iters=600):
    """Compute the general anomaly by solving the universal Kepler
    function using the Newton's method 

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    x : float
        the universal anomaly after time t [km^0.5]
    dt : float
        time since x = 0 [days]
    ro : np.array
        Initial radial position, i.e., when x=0 [AU]
    vro : np.array
        Initial radial velocity, i.e., when x=0 [AU]
    inv_a : float
        Reciprocal of the semimajor axis [1/AU]
    abs_tol : float, optional
        The aboluse error tolerance
    max_iters : int, optional
        Maximum allowable number of iterations, by default 500

    Returns
    -------
    float
        The universal anomaly (x) [AU^.5]
    """

    ratio = 1
    f = partial(_F, mu, r0, vr0, inv_a, dt)
    fprime = partial (_Fprime, mu, r0, vr0, inv_a)
    fprime2 = partial (_Fprime2, mu, r0, vr0, inv_a)

    X0 = sqrt(mu)*abs(inv_a)*dt
    x = X0
    N=5
    for i in range(0,max_iters):
        if isclose(ratio, 0, rel_tol=0, abs_tol=abs_tol):
            logger.debug(f"The laguerre method in Universal variables converged with {i} iterations")
            return x
        den1 = np.sqrt(np.abs(pow(N-1,2)*pow(fprime(x),2)-N*(N-1)*f(x)*fprime2(x)))
        if fprime(x)>0 :
            ratio = N*f(x)/(fprime(x)+den1)
        else:
            ratio = N*f(x)/(fprime(x)-den1)
        x = x - ratio
    logger.error(f'Universal Kepler equation not converged with Laguerre with X0: {X0}, root: {x} error: {ratio} F(x)={f(x)}') 
    raise NoConvergenceError(x, max_iters, max_iters, X0)

LINEAR_GRID = list(np.linspace(2.5,4,16,endpoint=True))

def kepler_U(mu, dt, ro, vro, inv_a, nMax=500):
    """Compute the general anomaly by solving the universal Kepler
    function  using the Newton's method

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    x : float
        the universal anomaly after time t [km^0.5]
    dt : float
        time since x = 0 [days]
    ro : np.array
        Initial radial position, i.e., when x=0 [AU]
    vro : np.array
        Initial radial velocity, i.e., when x=0 [AU]
    inv_a : float
        Reciprocal of the semimajor axis [1/AU]
    nMax : int, optional
        Maximum allowable number of iterations, by default 500

    Returns
    -------
    float
        The universal anomaly (x) [AU^.5]
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
    
    #return ku.kepler_U(mu, x, dt, ro, vro, inv_a, nMax)
    x = solve_kepler_universal_laguerre(mu, dt, ro, vro, inv_a, max_iters=500)
    return x
   
def calc_f_g(mu, x, t, ro, inv_a):
    """Calculates the Lagrange f and g coefficients starting from the initial
    position r0 (radio vector from the dinamical center (normally the Sun)
    and the elapsed time t)

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    x : float
        the universal anomaly after time t [km^0.5]
    t : float
        the time elapsed since ro (days)
    ro : np.array
        the radial position at time to [AU]
    inv_a : float
        reciprocal of the semimajor axis [1/AU]

    Returns
    -------
    tuple
        A tuple with f and g coefficients, i.e.,  (f,g)
    """
    z = inv_a*pow(x,2)
    f = 1 - pow(x,2)/ro*stump_C(z)    
    g = t - 1/sqrt(mu)*pow(x,3)*stump_S(z)
    return f, g 


def calc_fdot_gdot(mu, x, r, ro, inv_a) :
    """Calculates the time derivatives of Lagrange coefficients
    f and g coefficients.

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    x : float
        the universal anomaly after time t [AU^0.5]
    r : np.array
        the radial position (radio vector) after time t [AU]
    ro : np.array
        the radial position (radio vector) at time to [AU]
    inv_a : float
        reciprocal of the semimajor axis [1/AU]

    Returns
    -------
    tuple
        a tuple with fdot and gdot, i.e., (fdot, gdot)
    """

    z = inv_a*pow(x,2)
    #%...Equation 3.69c:
    fdot = sqrt(mu)/r/ro*(z*stump_S(z) - 1)*x
    # %...Equation 3.69d:
    gdot = 1 - pow(x,2)/r*stump_C(z)
    return fdot, gdot

def calc_f(p, X, r0, sigma0, inv_a, f0):
    """ Compute the true anomaly from Universal anomaly plus
    another data according to the formalul in Conway pg. 40

    Parameters
    ----------
    p : float 
        Semi Latus Rectum [AU]
    X : float
        Universal Anomaly
    r0 : float
        Modulus of the initial radio vector
    sigma0 : float
        r0*vr0/sqrt(mu)
    inv_a : float
        Reciprocal of the semimajor axis 
    f0 : float
        Initial true anomaly [radians]

    Returns
    -------
    float
        True anomaly [radians]
    """
    z = X/2
    alphaz_2 = inv_a*z*z
    C = stump_C(alphaz_2)
    S = stump_S(alphaz_2)
    num = z*np.sqrt(p)*(1-alphaz_2*S)
    den1 = r0*(1-alphaz_2*C)
    den2 = sigma0*z*(1-alphaz_2*S)
    f_f0_div2 = np.arctan2(num, den1+den2)
    f = 2*f_f0_div2+f0
    f = norm_rad(f)
    if isclose(f, TWOPI, rel_tol=0, abs_tol=1e-5):
        return 0.0 
    else :
        return f



def calc_rv_from_r0v0(mu, r0_xyz, r0dot_xyz, dt, f0=None):
    """This function computes the state vector (r_xyz, rdot_xyz) from the
    initial state vector (r0_xyz, r0dot_xyz) and after the elapsed time (dt)
    Internally uses the universal variables and the lagrange coefficients.
    Although according to the book, this is used in the perifocal plane 
    (i.e. the orbital plane), in the enckle method I used in the ecliptic 
    plane and it works. It may be becasue at the end the the size of the
    orbital plane does not change only it is rotated according to the
    Gauss angles.

    To solve the universal variables equation, the Laguerre method is used. It works
    very nice with a mininmun number of non-convergences. It is better than Netwon's method

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    r0_xyz : np.array
        Initial position vector at t=0 [AU]
    r0dot_xyz : np.array
        Initial position vector at t=0 [AU]
    dt : float
        Elapsed time from t=t0 [days]
    f0 :  float, optional
        Initial true anomaly (needed to compute the true anomaly at t=dt)

    Returns
    -------
    tuple
        A tuple (r_xyz, rdot_xyz, h_xyz, f) where:
            r_xyz: Final position vector at t=dt [AU]
            rdot_xyz: Final position vector at t=dt [AU/days]
            h_xyz: Angular momentum vector at t=dt 
            f : True anommaly (if f0 is provided) after at t=dt
    """
    #The norm of the inital radio vector and velocity vector is calculated
    r0 = norm(r0_xyz)
    v0 = norm(r0dot_xyz)

    #The initial radia velocity is calculated
    vr0 = np.dot(r0_xyz, r0dot_xyz)/r0

    # Reciprocal of the semimajor axis (from the energy equation):
    #   alpha < 0 for Hyperbolic
    #   alpha = 0 for Parabolic
    #   alpha > 0 for Elliptical

    alpha = 2/r0 - pow(v0,2)/mu
    # For some parabolic comets the calculated alpha is around 1e-15. Those
    # values prevents the laguerre method from converging. In those cases,
    # the alpha value is set to 0.0
    if isclose(alpha, 0.0, rel_tol=0, abs_tol=1e-13):
        logger.debug(f"Correcting the value of alpha from {alpha} to 0.0, the object should have e=1.0 (parabolic)")
        alpha = 0.0

    # The kepler equation is solved to obtain the Universal anomaly
    X = solve_kepler_universal_laguerre (mu, dt, r0, vr0, alpha, abs_tol=1.0e-10, max_iters=600)
                
    #Compute the f and g functions:
    f, g = calc_f_g(mu, X, dt, r0, alpha)

    #Compute the final position vector:
    r_xyz = f*r0_xyz + g*r0dot_xyz

    #Compute the magnitude of r_xyz    
    r = norm(r_xyz)

    #Compute the derivatives of f and g:
    fdot, gdot = calc_fdot_gdot(mu, X , r, r0, alpha)

    #Compute the final velocity vector
    rdot_xyz = fdot*r0_xyz + gdot*r0dot_xyz

    # The angular momentum
    h_xyz = np.cross(r_xyz, rdot_xyz)   

    if f0 is not None :
        # The norm of the angular momentum
        h = np.linalg.norm(h_xyz)
        # Semi-Latus Rectum
        p = pow(h,2)/mu
        sigma0 = r0*vr0/sqrt(mu)
        f = calc_f(p, X, r0, sigma0, alpha, f0)
    else :
        f = None
    
    return r_xyz, rdot_xyz, h_xyz, f

if __name__ == "__main__":
    None


 