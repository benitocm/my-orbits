""" This module contains functions related to Lagrange Coefficients and universal variables
The source comes mostly from the book 'Orbital Mechanics for Engineering Students'
"""
# Standard library imports
import logging


# Third party imports
import numpy as np
from numpy import sin, cos, sqrt, cosh, sinh, sqrt, abs

# Local application imports
from myorbit.util.general import KahanAdder, NoConvergenceError
from myorbit.init_config import STUMPFF_ABS_TOL, STUMPFF_METHOD
logger = logging.getLogger(__name__)

def calc_stumpff(x, atol=STUMPFF_ABS_TOL, max_iters=200):
    """Computes the values for the Stumpff functions c1, c2, c3 

    Parameters
    ----------
    x : float
        The value of x
    atol : float, optional
        Absolute accuracy, by default 1.e-15        
    max_iters : int, optional
        Maximum number of iterations, by default 200

    Returns
    -------
    tuple (c1, c2, c3)
        The three stumpff values [float]
        
    Raises
    ------
    NoConvergenceError
        When the series does not converge (this should no happen)
        
    """
    
    if STUMPFF_METHOD == 0:
        return calc_stumpff_as_series(x,atol, max_iters)
    else :
        return calc_stumpff_exact(x)

def calc_stumpff_as_series(x, atol=STUMPFF_ABS_TOL, max_iters=200):    
    """Computes the values for the Stumpff functions C1, C2, C3 summing up
    an infinite series

    Parameters
    ----------
    x : float
        The value of x
    atol : float, optional
        Absolute accuracy, by default 1.e-15        
    max_iters : int, optional
        Maximum number of iterations, by default 200

    Returns
    -------
    tuple (C1, C2, C3)
        The three stumpff values [float]
        
    Raises
    ------
    NoConvergenceError
        When the series does not converge (this should no happen)
        
    """
    c1, c2, c3 = KahanAdder(), KahanAdder() , KahanAdder()
    to_add = 1.0
    for n in range(1, max_iters):
        c1.add(to_add)
        to_add = to_add/(2.0*n)
        c2.add(to_add)
        to_add = to_add/(2.0*n+1.0)
        c3.add(to_add)
        to_add = to_add*-x
        if c1.converged(atol) and c2.converged(atol) and c3.converged(atol): 
            return c1.result(), c2.result(), c3.result()
    logger.error(f"Stumpff functionis Not converged after {n} iterations")
    raise NoConvergenceError(c1.result(), n, n, "Stumpff functions does not converge")

        
def calc_stumpff_exact(x):    
    """Computes the values for the Stumpff functions C1, C2, C3 according to
    its definition

    Parameters
    ----------
    x : float        

    Returns
    -------
    tuple (C1, C2, C3)
        The three stumpff values [float]
    """
    if abs(x) <= 1.e-14:
        c1 = 1.
        c2 = 1/2
        c3 = 1/6
    elif x < 0 :
        xr = np.sqrt(-x)
        c1 = np.sinh(xr)/xr
        c2 = (np.cosh(xr)-1)/-x
        c3 = (np.sinh(xr)-xr)/(-x*xr)
    elif x > 0 :
        xr = np.sqrt(x)
        c1 = np.sin(xr)/xr
        c2 = (1-np.cos(xr))/x
        c3 = (xr-np.sin(xr))/(x*xr)
    return c1, c2, c3
    
    
    
if __name__ == "__main__":
    None
       
