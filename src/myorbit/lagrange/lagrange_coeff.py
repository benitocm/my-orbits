""" This module contains functions related to Lagrange Coefficients and universal variables
The source comes mostly from the book 'Orbital Mechanics for Engineering Students'
"""
# Standard library imports
from functools import partial
import logging

# Third party imports
import numpy as np
from numpy import sin, cos, sqrt, cosh, sinh, sqrt, abs
from numpy.linalg import norm
from scipy.optimize import newton
from numba import jit

# Local application imports
from myorbit.util.general import pow, NoConvergenceError, calc_ratio
from myorbit.util.timeut import norm_rad  


from pathlib import Path
CONFIG_INI=Path(__file__).resolve().parents[3].joinpath('conf','config.ini')
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read(CONFIG_INI)

LAGUERRE_ABS_TOL = float(cfg.get('general','laguerre_abs_tol'))

logger = logging.getLogger(__name__)

      
@jit(nopython=True)   
def stumpff_C(z) :
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

@jit(nopython=True)
def stumpff_S(z) :    
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
        return (sz - sin(sz))/(z*sz)
    elif z < 0 :
        sz = sqrt(-z)         
        return  (sinh(sz) - sz)/(-z*sz)
    else :
        return 1/6    

@jit(nopython=True)
def _F(mu, r0, vr0, inv_a, dt, x):
    """Compute the Kepler equation in Universal 
    variables formulation

    Parameters
    ----------
    mu : float
         Gravitational parameter [AU^3/days^2][description]
    r0 : float
        Modulus of the initial radio vector
    vr0 : float
        Modulus of the initial radial velocity
    inv_a : float
        Reciprocal of the semimajor axis 
    dt : float
        The elapsed time from t=0 [days]
    x : float
        The universal anomaly

    Returns
    -------
    [type]
        [description]
    """
    z = x*x
    C = stumpff_C(inv_a*z)
    S = stumpff_S(inv_a*z)
    return  (r0*vr0/sqrt(mu))*z*C + (1 - inv_a*r0)*z*x*S + r0*x - sqrt(mu)*dt

@jit(nopython=True)
def _Fprime(mu, r0, vr0, inv_a, x):
    """Compute the first derivative of the Kepler equation in Universal 
    variables formulation

    Parameters
    ----------
    mu : float
         Gravitational parameter [AU^3/days^2][description]
    r0 : float
        Modulus of the initial radio vector
    vr0 : float
        Modulus of the initial radial velocity
    inv_a : float
        Reciprocal of the semimajor axis 
    x : float
        The universal anomaly

    Returns
    -------
    float
        The first derivative of the Kepler universal equation
    """
    z = x*x
    C = stumpff_C(inv_a*z)
    S = stumpff_S(inv_a*z)
    return (r0*vr0/sqrt(mu))*x*(1 - inv_a*z*S) + (1 - inv_a*r0)*z*C + r0

@jit(nopython=True)
def _Fprime2(mu, r0, vr0, inv_a, x):
    """Compute the second derivative of the Kepler equation in Universal 
    variables formulation

    Parameters
    ----------
    mu : float
         Gravitational parameter [AU^3/days^2][description]
    r0 : float
        Modulus of the initial radio vector
    vr0 : float
        Modulus of the initial radial velocity
    inv_a : float
        Reciprocal of the semimajor axis 
    x : float
        The universal anomaly

    Returns
    -------
    float
        The second derivative of the Kepler universal equation
    """
    z = x*x
    C = stumpff_C(inv_a*z)
    S = stumpff_S(inv_a*z)
    return r0*vr0/sqrt(mu) - (inv_a*r0*vr0/sqrt(mu))*pow(x,2)*C + (1-inv_a*r0)*x-(1-inv_a*r0)*inv_a*pow(x,3)*S



@jit(nopython=True)
def calc_true_anomaly(p, X, r0, sigma0, inv_a, f0):
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
    C = stumpff_C(alphaz_2)
    S = stumpff_S(alphaz_2)
    num = z*np.sqrt(p)*(1-alphaz_2*S)
    den1 = r0*(1-alphaz_2*C)
    den2 = sigma0*z*(1-alphaz_2*S)
    f_f0_div2 = np.arctan2(num, den1+den2)
    f = 2*f_f0_div2+f0
    f = norm_rad(f)
    return f

@jit(nopython=True)  
def calc_f_g(mu, X, dt, r0, inv_a):
    """Calculates the Lagrange f and g coefficients starting from the initial
    position r0 (radio vector from the dinamical center (normally the Sun)
    and the elapsed time t)

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    X : float
        the universal anomaly after time t [km^0.5]
    dt : float
        the time elapsed since ro (days)
    r0 : np.array
        the radial position at time to [AU]
    inv_a : float
        reciprocal of the semimajor axis [1/AU]

    Returns
    -------
    tuple
        A tuple with f and g coefficients, i.e.,  (f,g)
    """
    z = inv_a*X*X
    f = 1 - (X*X/r0)*stumpff_C(z)    
    g = dt - (1/sqrt(mu))*X*X*X*stumpff_S(z)
    return f, g 

@jit(nopython=True)  
def calc_fdot_gdot(mu, x, r, r0, inv_a) :
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
    r0 : np.array
        the radial position (radio vector) at time to [AU]
    inv_a : float
        reciprocal of the semimajor axis [1/AU]

    Returns
    -------
    tuple
        a tuple with fdot and gdot, i.e., (fdot, gdot)
    """
    
    z = inv_a*pow(x,2)
    # Equation 3.69c in Orbital Mechanics for Engineering
    fdot = sqrt(mu)/r/r0*(z*stumpff_S(z) - 1)*x
    # Equation 3.69d  in Orbital Mechanics for Engineering
    gdot = 1 - pow(x,2)/r*stumpff_C(z)
    return fdot, gdot

def solve_kepler_eq (mu, X0, r0, vr0, inv_a, dt):
    """Compute the universal anomaly by solving the universal Kepler
    function using Newton's  method. 

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    X0 : float
        The inital value for universal anomaly  [km^0.5]
    r0 : np.array
        Initial radial position, i.e., when X=0 [AU]
    vr0 : np.array
        Initial radial velocity, i.e., when X=0 [AU]
    inv_a : float
        Reciprocal of the semimajor axis [1/AU]
    dt : float
        time since X = 0 [days]        

    Returns
    -------        
    tuple (X, iters, ratio, max_iters):
        X: The universal anomaly (x) [AU^.5]
        root: A structure retuned by the newton's method
    Raises
    ------
    NoConvergenceError
        When the method to find the root of the Kepler's equation does not converge        
    """
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

    X, root = newton(f, X0, fprime=fprime, fprime2=fprime2, tol=1e-09, maxiter=700,  full_output=True, disp=False)
    if not root.converged:        
       logger.error(f'Universal Kepler equation not converged with root:{root}') 
       raise NoConvergenceError(x, root.function_calls, root.iterations, X0)
    logger.debug(f'Converged in {root.iterations} iterations and {root.function_calls} function_calls for X0={X0}, ro={r0}, vro={vr0}, inv_a={inv_a}, dt={dt}  Not converged with root:{root}') 
    #TODO to return adapt to what it is returned in the laguerre's method.
    return X, root 

@jit(nopython=True)  
def solve_kepler_universal_laguerre (mu, X0, dt, r0, vr0, inv_a, abs_tol=LAGUERRE_ABS_TOL, max_iters=600):
    """Compute the universal anomaly by solving the universal Kepler
    function using the Laguerre method. This function has been modified
    to be jited with Numba

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    X0 : float
        The inital value for universal anomaly  [km^0.5]
    dt : float
        time since X = 0 [days]
    r0 : np.array
        Initial radial position, i.e., when X=0 [AU]
    vr0 : np.array
        Initial radial velocity, i.e., when X=0 [AU]
    inv_a : float
        Reciprocal of the semimajor axis [1/AU]
    abs_tol : float, optional
        The absolute error tolerance
    max_iters : int, optional
        Maximum allowable number of iterations, by default 600

    Returns
    -------        
    tuple (X, iters, ratio, max_iters):
        X: The universal anomaly (x) [AU^.5]
        iters: The number of iterations when the algorithm converged. If there is no convergence, -1
        ratio: The ratio obtained by the convergence algorithm
        max_iters : Maximum number of iterations
    """
    ratio = 1
    x = X0
    N=5
    for i in range(0,max_iters):
        if np.abs(ratio) <= abs_tol :
            return x, i, ratio, max_iters
        else :
            f_at_x = _F (mu, r0, vr0, inv_a, dt, x)
            fprime_at_x = _Fprime( mu, r0, vr0, inv_a, x)
            fprime2_at_x = _Fprime2( mu, r0, vr0, inv_a ,x)
            ratio = calc_ratio(N, f_at_x, fprime_at_x, fprime2_at_x)
            x = x - ratio
    return x, -1, ratio, max_iters

LINEAR_GRID = list(np.linspace(2.5,4,16,endpoint=True))

def calc_rv_from_r0v0(mu, r0_xyz, r0dot_xyz, dt, f0=None):
    """Computes the state vector (r_xyz, rdot_xyz) from the
    initial state vector (r0_xyz, r0dot_xyz) and after the elapsed time (dt)
    Internally uses the universal variables and the lagrange coefficients.
    Although according to the book, this is used in the perifocal plane 
    (i.e. the orbital plane), in the enckle method I have used it in the ecliptic 
    plane and it works. It may be because, at the end, the the size of the
    orbital plane does not change only it is rotated according to the
    Gauss angles.

    To solve the universal variables equation, the Laguerre method is used. It works
    very nice with a mininmun number of non-convergences. In my experience, it works
    better than Netwon's method.

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    r0_xyz : np.array[3]
        Initial position vector at t=0 [AU]
    r0dot_xyz : np.array[3]
        Initial velocity vector at t=0 [AU]
    dt : float
        Elapsed time from t=t0 [days]
    f0 :  float, optional
        Initial true anomaly (needed to compute the true anomaly at t=dt)
        If its provided, the final true anomaly is calculated.
        
    Returns
    -------
    tuple (r_xyz, rdot_xyz, h_xyz, f) where:
        r_xyz: is a np.array[3] that contains the final position, i.e., the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane (perifocal frame) [AU]
        rdot_xyz: is a np.array[3] that contains the final velocity vector (cartesian) of the body
            with respect to the orbital plane (perifocal frame) [AU/days]
        h_xyz : h_xyz: Angular momentum vector at t=dt 
        f : True anommaly [radians] (if f0 is provided) after at t=dt
        
    Raises
    ------
    NoConvergenceError
        When the method to find the root of the Kepler's equation does not converge
        
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
    
    if np.abs(alpha) < 1.e-13 :
        logger.debug(f"Correcting the value of alpha from {alpha} to 0.0, the object should have e=1.0 (parabolic)")
        alpha = 0.0

    X0 = sqrt(mu)*abs(alpha)*dt
    # The kepler equation is solved to obtain the Universal anomaly        
    X, iters, ratio, max_iters = solve_kepler_universal_laguerre (mu, X0, dt, r0, vr0, alpha)
    if iters == -1 :
        logger.error(f'Universal Kepler equation not converged with Laguerre with X0: {X0}, root: {X} error: {ratio}') 
        raise NoConvergenceError(X, max_iters, max_iters, X0)
    else :
        logger.debug(f"The laguerre method in Universal variables converged with {iters} iterations")
          
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
        f = calc_true_anomaly(p, X, r0, sigma0, alpha, f0)
    else :
        f = None
    
    return r_xyz, rdot_xyz, h_xyz, f


if __name__ == "__main__":
    None


 