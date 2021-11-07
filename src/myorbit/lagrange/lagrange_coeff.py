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



# Local application imports
from myorbit.util.general import pow
from myorbit.util.general import pow, NoConvergenceError
import  myorbit.lagrange.kepler_u as ku
from myorbit.util.constants import *

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

def _F(mu, ro, vro, inv_a, dt, x):
    z = x*x
    C = stump_C(inv_a*z)
    S = stump_S(inv_a*z)
    return  ro*vro/sqrt(mu)*z*C + (1 - inv_a*ro)*z*x*S + ro*x - sqrt(mu)*dt

def _Fprime(mu, ro, vro, inv_a, x):
    z = x*x
    C = stump_C(inv_a*z)
    S = stump_S(inv_a*z)
    return ro*vro/sqrt(mu)*x*(1 - inv_a*z*S) + (1 - inv_a*ro)*z*C + ro


def _Fprime2(mu, ro, vro, inv_a, x):
    z = x*x
    C = stump_C(inv_a*z)
    S = stump_S(inv_a*z)
    return ro*vro/sqrt(mu) - (inv_a*ro*vro/sqrt(mu))*pow(x,2)*C + (1-inv_a*ro)*x-(1-inv_a*ro)*2*inv_a*pow(x,3)*S

    

def solve_kepler_eq(mu, ro, vro, inv_a, dt):
    # The first 5 parameters of F are bounded, so we end up with f(x)
    # So f(x) = 0 is the equation that is desired to solve (the kepler equation)
    # for universal anomaly
    f = partial(_F, mu, ro, vro, inv_a, dt)

    # The first 4 parameter of Fprime are bounded, so we end up with fprime(x)
    # According to the newton method, it is better if the first derivative of f is available
    fprime = partial (_Fprime, mu, ro, vro, inv_a)

    fprime2 = partial (_Fprime2, mu, ro, vro, inv_a)

    # The inital value for the universal anomaly is calculated.
    X0 = np.sqrt(mu)*np.abs(inv_a)*dt

    # Kepler equation is solved
    x, root = newton(f, X0, fprime=fprime, fprime2=fprime2, tol=1e-09, maxiter=500,  full_output=True, disp=False)
    if not root.converged:        
       logger.error(f'Not converged with root:{root}') 
       raise NoConvergenceError(x, root.function_calls, root.iterations, X0)
    logger.info(f'Converged in {root.iterations} iterations and {root.function_calls} function_calls for X0={X0}, ro={ro}, vro={vro}, inv_a={inv_a}, dt={dt}  Not converged with root:{root}') 
    return x, root 

def kepler_U_prv(mu, x , dt, ro, vro, inv_a, nMax=500):
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
    nMax : int, optional
        Maximum allowable number of iterations, by default 500

    Returns
    -------
    float
        The universal anomaly (x) [AU^.5]
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
    x = sqrt(mu)*abs(inv_a)*dt
    return ku.kepler_U(mu, x, dt, ro, vro, inv_a, nMax)
   
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

def calc_rv_from_r0v0(mu, r0_xyz, r0dot_xyz, dt):
    """This function computes the state vector (R,V) from the
    initial state vector (R0,V0) and after the elapsed time.
    Internally uses the universal variables and the lagrange coefficients.
    Although according to the book, this is used in the perifocal plane 
    (i.e. the orbital plane), in the enckle method I used in the ecliptic 
    plane and it works. It may be becasue at the end the the size of the
    orbital plane does not change only it is rotated according to the
    Gauss angles.

    Parameters
    ----------
    mu : float
        Gravitational parameter [AU^3/days^2]
    r0_xyz : np.array
        initial position vector at t0 [AU]
    r0dot_xyz : np.array
        initial position vector at t0 [AU]
    dt : float
        Elapsed time from t=t0 [days]

    Returns
    -------
    tuple
        A tuple (r_xyz, rdot_xyz) where:
            r_xyz: Final position vector after dt (AU)
            rdot_xyz: Final position vector after dt (AU/days)

    """
    #The norm of the inital radio vector and velocity vector is calculated
    r0 = norm(r0_xyz)
    v0 = norm(r0dot_xyz)

    #The initial radia velocity is calculated
    vr0 = np.dot(r0_xyz, r0dot_xyz)/r0

    # Reciprocal of the semimajor axis (from the energy equation):
    alpha = 2/r0 - pow(v0,2)/mu

    # The kepler equation is solved to obtain the Universal anomaly
    x, _ = solve_kepler_eq(mu, r0, vr0, alpha, dt)    

    #x = kepler_U(mu, t, r0, vr0, alpha)

    #Compute the f and g functions:
    f, g = calc_f_g(mu, x, dt, r0, alpha)

    #Compute the final position vector:
    r_xyz = f*r0_xyz + g*r0dot_xyz

    #Compute the magnitude of r_xyz    
    r = norm(r_xyz)

    #Compute the derivatives of f and g:
    fdot, gdot = calc_fdot_gdot(mu, x, r, r0, alpha)

    #Compute the final velocity vector
    rdot_xyz = fdot*r0_xyz + gdot*r0dot_xyz

    return r_xyz, rdot_xyz, np.cross(r_xyz, rdot_xyz)   

def calc_eccentricity_vector(r_xyz, rdot_xyz, h_xyz,mu):
    return  (np.cross(rdot_xyz,h_xyz) - (mu*r_xyz/np.linalg.norm(r_xyz)))/mu

def angle_between_vectors(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return  np.arccos(dot_product)    


def test1() :
    mu = 398600
    R0 = np.array([7000, -12124, 0])
    V0 = np.array([2.6679, 4.6210, 0])
    h0_xyz = np.cross(R0,V0)
    t = 3600
    r_xyz, rdot_xyz, h_xyz= calc_rv_from_r0v0(mu,R0, V0, t)
    e0_xyz = calc_eccentricity_vector(R0, V0, h0_xyz, mu)
    e_xyz = calc_eccentricity_vector(r_xyz, rdot_xyz, h_xyz,mu)
    print (h0_xyz, h_xyz)
    print (np.linalg.norm(e0_xyz), np.linalg.norm(e_xyz))
    print (f"True Anomaly at t0 :{angle_between_vectors(e0_xyz, R0)}")
    print (f"True Anomaly at t :{angle_between_vectors(e_xyz, r_xyz)}")
    print ("R: ",r_xyz)
    print ("V: ",rdot_xyz)

def test2():
    mu = 398600
    ro =  13999.691
    vro = -2.6678
    inv_a = 7.143e-05
    
    vro 

    f = partial(_F, mu, ro, vro, inv_a, dt)

    # The first 4 parameter of Fprime are bounded, so we end up with fprime(x)
    # According to the newton method, it is better if the first derivative of f is available
    fprime = partial (_Fprime, mu, ro, vro, inv_a)

    fprime2 = partial (_Fprime2, mu, ro, vro, inv_a)    
    R0 = np.array([7000, -12124, 0])
    V0 = np.array([2.6679, 4.6210, 0])
    h0_xyz = np.cross(R0,V0)




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


 