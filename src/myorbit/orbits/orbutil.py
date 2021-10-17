""" This module contains utility functions to keplerian orbit calculations
"""
# Standard library imports
from functools import partial
from math import isclose
import logging

# Third party imports
import numpy as np
from numpy.linalg import norm
# Using Newton-Ramson method
from scipy.optimize import newton, bisect
from scipy.integrate import solve_ivp    

# Local application imports
from myorbit import coord as co
from myorbit.util.timeut  import  MDJ_J2000, JD_J2000, CENTURY, mjd2jd
from myorbit.util.general import pow
from myorbit.planets import h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000
from myorbit.util.constants import *

logger = logging.getLogger(__name__)

def do_iterations(func, x_0, abs_tol = 1e-08, max_iter=50):
    """General function to iterate mathematical functions of one variable

    Parameters
    ----------
    func : function
        The mathematical funtion of one variable to iterate
    x_0 : float
        Initial value of the variable
    abs_tol : float, optional
        [description], by default 1e-08
    max_iter : int, optional
        Maximum number of iterations, by default 50

    Returns
    -------
    tuple (boolean, float, int)
        The result of the iteration where:
            boolean param, indicates whether the solution converged or not
            float param, is the final value of the variable
            n, is the last iteration done
    """
    x = x_0
    for n in range(max_iter):
        new_value = func(x)
        if isclose(new_value, x, abs_tol = abs_tol) :
            return (True,new_value,n)
        else :
            x = new_value
    return (False,new_value,n)

def solve_ke(e, func_e_anomaly,  m_anomaly):
    """Solves the kepler equation, i.e., calculates the excentric anomaly

    Parameters
    ----------
    e : float
        The eccentricity [0,1]
    func_e_anomaly : function
        Function to calculate the excentric anomaly used to iterate
    m_anomaly : float
        Mean anomaly in angle units (rads)

    Returns
    -------
    float
        The value fo the excentric anomaly (if converged). Otherwise, return None        
    """

    f = partial(func_e_anomaly,e,m_anomaly)
    res = do_iterations(f,m_anomaly,100)
    if not res[0] :
        logger.error("Not converged")
        return 
    else :
        logger.debug(f"Converged in {res[2]} iterations wih result {np.rad2deg(res[1])} degrees")
        return res[1]

def solve_ke_newton(e, func_e_anomaly, m_anomaly, e_anomaly_0=None):
    """ Solves the kepler equation, i.e., calculates the excentric anomaly by using
     the Newton-Ranson method

    Parameters
    ----------
    e : float
        eccentricity [0,1]
    func_e_anomaly : function
        function to calculate the excentric anomaly used to iterate
    m_anomaly : float
        Mean anomaly in angle units (radians)
    e_anomaly_0 : float, optional
        Initial value to start the iteration, by default None

    Returns
    -------
    float
        The value fo the excentric anomaly (if converged). Otherwise, return None        
    """
  
    f = partial(func_e_anomaly , e ,m_anomaly)
    x, root = newton(lambda x : f(x) - x, e_anomaly_0, fprime=None, tol=1e-12, maxiter=50, fprime2=None, full_output=True)    
    logger.error(root)
    if not root.converged :
        logger.error(f"Not converged: {root}")
    return x

def solve_ke_bisect(e,m_anomaly,func_e_anomaly):
    """[summary]

    Parameters
    ----------
    e : [type]
        [description]
    m_anomaly : [type]
        [description]
    func_e_anomaly : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    m_rad = np.deg2rad(m_anomaly)    
    f = partial(func_e_anomaly,e,m_rad)    
    x = bisect(f, 0, 0.25, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, full_output=False, disp=True)
    return x

    """
    Computes the acceleration based on the corresponding GM and the
    radio vector.
    Returns a vector
    """

def accel(gm, r):
    """Computes the acceleration based on the corresponding GM and the
    radio vector (distance to the Sun)

    Parameters
    ----------
    gm : float
       The product G*M (Gravity constant * Sun Mass)
    r : float
        Distance to the Sun

    Returns
    -------
    float
        The acceleration 
    """
    return gm*r/pow(norm(r),3) 

    """
    Computes the acceleration vector for a minor body in the solar system at one point
    time.

    Args:
        t :  point in time in Modified Julian days
        h_xyz_eclip_body : Heliocentric Ecliptic Cartesian coordinates (J2000) of the minor body (as numpy vector of 3)

    Returns :
        The acceleration as J2000 ecliptic coordinate vector (numpy vector of 3) [AU/d**2]
    """


def calc_perturbed_accelaration(t_mjd, h_xyz_eclip_body ) :
    """Computes the acceleration vector for a minor body in the solar system at one point
    time. The perturbation caused by the planets (including Pluto) to the minor body 
    are taking into account

    Parameters
    ----------
    t_mjd : float
        Point in time as Modified Julian Day 
    h_xyz_eclip_body : np.array[3]
        Heliocentric cartesian coordinates of the minor body

    Returns
    -------
    float
        The acceleration
    """

    # The century T corresponding to the time t. Also used to calculate
    # precession matrix
    T = (t_mjd - MDJ_J2000)/CENTURY    
    # desired equinox is J2000, so T_desired is 0
    T_desired = (JD_J2000 - JD_J2000)/CENTURY
    mtx_prec = co.mtx_eclip_prec(T, T_desired)
    acc = 0
    #for pl_name in filter(lambda x : (x != 'Sun') and (x != 'Pluto'), GM_by_planet.keys()) :
    for pl_name in filter(lambda x : (x != 'Sun') , GM_by_planet.keys()) :
        if pl_name == 'Pluto':
            h_xyz_eclipt_planet = h_xyz_eclip_pluto_j2000(mjd2jd(t_mjd))
        else :
            # Planetary position (ecliptic and equinox of J2000)
            h_xyz_eclipt_planet = mtx_prec.dot(h_xyz_eclip_eqxdate(pl_name, mjd2jd(t_mjd)))

        h_xyz_planet_body = h_xyz_eclip_body - h_xyz_eclipt_planet
        # Direct accelaration
        acc += accel(GM_by_planet[pl_name], h_xyz_planet_body)
        # Indirect acceletration
        acc += accel(GM_by_planet[pl_name], h_xyz_eclipt_planet)
    return -acc    



def do_integration(fun_t_y, y0 , t_begin, t_end, t_samples):
    sol = solve_ivp(fun_t_y, (t_begin,t_end), y0, t_eval=t_samples, rtol = 1e-12)  
    if sol.success :
        return sol
    else :
        logger.warn("The integration was failed: "+sol.message)
