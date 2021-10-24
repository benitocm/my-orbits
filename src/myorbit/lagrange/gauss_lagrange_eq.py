"""
This module contains functions related to orbit calculations
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

from functools import partial
from math import isclose
import sys

# Third party imports
import pandas as pd
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, arctan, sqrt,cosh,sinh,deg2rad,rad2deg, sqrt, arcsinh
from toolz import pipe, compose, first, valmap
from myastro.util import pow
from math import  isclose, fmod

from myastro.log import get_logger
logger = get_logger(__file__.split('/')[-1])

def r1 (sigma, rho_0, rho):
    """
     Auxiliary function r1(rho) = (sigma/(rho_0-rho))^(1/3) of the 
     Gauss-Lagrangian equation    
     
     Args:
        sigma    Auxiliary quantity
        rho_0    Auxiliary quantity
        rho      Geocentric distance
        
     Returns:
        Heliocentirc distance
    """
    return np.cbrt(sigma/(rho_0-rho))
 

def r2 (gamma, R, rho):
    """
     function r2(rho) = sqrt( (rho-gamma*R)^2+R^2-(gamma*R)^2 ) of
     Gauss-Lagrangian equation    
     
     Args:
         gamma    Angle Sun-Earth-Body
         R        Earth-Sun distance
         rho      Geocentric distance
         
     Returns:
        Heliocentirc distance
    """

    gR = gamma*R
    return sqrt (pow(rho-gR,2) + pow(R,2) - pow(gR,2))
        
def bisect(gamma, R, rho_0, sigma, rho_1, rho_2):
    """
    Solution of the Gauss-Lagrangian equation (r1(rho)=r2(rho)) 
    using bisection in a given interval.
    
    Note:
    rho_1 and rho_2 may themselves be roots of the Gauss-Lagrangian 
    equation provided that an independent root rho exists within the
    interval ]min(rho_1,rho_2), max(rho_1,rho_2)[ and that r1<=r2 between
    rho_1 and rho and r1>=r2 between rho and rho_2. This may imply that 
    rho_2 < rho_1!
       
    Args:
        gamma    Angle Sun-Earth-Body
        R        Earth-Sun distance
        rho_0    Geocentric distance
        sigma    Auxiliary quantity
        rho_1    Bound of search interval (r1-r2 <= 0.0)
        rho_2    Bound of search interval (r1-r2 >= 0.0)
        
    Returns:
        rho:      Solution        
    """
    max_iter = 100
    for _ in range(max_iter):
        rho = (rho_1+rho_2)/2.0
        if r1(sigma, rho_0, rho) > r2(gamma, R, rho) :
            rho_2=rho
        else :
            rho_1=rho            
        if isclose(rho_1, rho_2, abs_tol=1e-9) :
            return rho
    logger.error(f"Not converged after {max_iter} iterations")
    return rho
 
    
def my_newton(gamma, R, rho_0, sigma, rho_start):
    """
    Solution of the Gauss-Lagrangian equation (r1(rho)=r2(rho)) 
    using Newton's iteration
    The Newton iteration is continued as long as monotonic convergence to 
    a solution is assured. Otherwise the abort flag is set.
    
    Args:
       gamma      Angle Sun-Earth-Body
       R          Earth-Sun distance
       rho_0      Geocentric distance
       sigma      Auxiliary quantity
       rho_1      Bound of search interval (r1-r2 <= 0.0)
       rho_2      Bound of search interval (r1-r2 >= 0.0)
       rho_start  Initial guess

    Returns
       rho        Solution      
       abort      Status code 
    
    """
    epsilon = 1e-8
    s = 1 if sigma > 0 else -1
    f=0
    df=1
    abort=1
    rho = rho_start
    max_iters = 100
    #% Newton iteration
    for _ in range(max_iters):
        #Check for improper initial conditions 
        if (s*(rho-gamma*R)<0) or (s*(rho-rho_0)>=0)  :
            logger.error(f"Improper initial conditions")
            break
        #Components of Gauss-Lagrange function
        r_a = r1(sigma, rho_0, rho)
        r_b = r2(gamma, R, rho)    

        # Gauss-Lagrange function and derivatives
        f   = r_a-r_b
        df  =  pow(r_a,4)/(3*sigma) - (rho-gamma*R)/r_b
        d2f = 4*pow(r_a,7)/(9*pow(sigma,2)) - pow(R,2)*(1-pow(gamma,2))/pow(r_b,3)
                
        #Check for success and/or abort
        if isclose(f, 0.0, abs_tol=epsilon) :
            abort = 0
            break
        
        # Check for abort condition based on gradient and curvature
        if  (s*df<=0) or (f*d2f<=0)  :
            logger.error(f"Abort condition based on gradient and curvature")
            break 
        
        # Update the solution        
        rho = rho-f/df 
    return rho, abort
    
def solve_equation (gamma, R, rho_0, sigma):
    """
    Finds the (up to three) solutions of the Gauss-Lagrangian 
    equation. Only positive solutions are returned. 
    
    Args:
       gamma: Cosine of elongation from the Sun at time of 2nd observation
           R: Earth-Sun distance at time of 2nd observation
       rho_0: 0th-order approximation of geocentric distance 
       sigma: Auxiliary quantity

    Returns
      rho_a: Geocentric distance at time of 2nd observation (1st solution)
      rho_b: Geocentric distance at time of 2nd observation (2nd solution)
      rho_c: Geocentric distance at time of 2nd observation (3rd solution)
          n: Number of solutions actually found
    """

    rho_min = 0.01 # Threshold for solutions
                   # of Gauss-Lagr. Eqn. [AU]
    #Sign of sigma
    s = 1 if sigma > 0 else -1
    
    # Initialization
    rho_a = 0
    rho_b = 0
    rho_c = 0

    # Check for single/triple solution
    if s*(rho_0 - gamma*R) <= 0 :
        # Single solution within [rho_d, rho_e]
        rho_d = rho_0 - s*pow(abs(sigma), 0.25)
        rho_e = rho_0 - sigma/pow(abs(gamma*R-rho_d)+R,3)
        rho_a = bisect (gamma, R, rho_0, sigma, rho_d, rho_e)
        n = 1
    else :
        # Triple solution possible
        # Locate root (c)
        if r1(sigma,rho_0,gamma*R) >= r2(gamma,R,gamma*R) :    
            rho_d = gamma*R-s*np.cbrt(sigma/(rho_0-gamma*R))
            rho_c = bisect(gamma, R, rho_0, sigma, rho_d,gamma*R)
            abort_c = 0
        else :
            rho_c, abort_c = my_newton(gamma, R, rho_0, sigma, gamma*R)   

        # Locate root (a)
        rho_d = rho_0 - sigma/pow(abs(rho_0-gamma*R)+R,3)
        rho_a,  abort_a = my_newton(gamma, R, rho_0, sigma, rho_d)
    
        # Locate root (b)
        if abort_a or abort_c :
            n=1
            if abort_a :
                rho_a=rho_c
        else :
            n=3
            rho_b = bisect(gamma, R, rho_0, sigma, rho_a, rho_c)
            #Sort solutions in ascending order
            if rho_a > rho_b :
                h = rho_a
                rho_a = rho_c
                rho_c = h    

    # Cyclic shift and elimination of negative and near-Earth solutions
    while ((rho_a < rho_min) and (n>1))  or (n==3)  :
        n -= 1
        h = rho_a
        rho_a = rho_b
        rho_b = rho_c
        rho_c = h        
    return rho_a, rho_b, rho_c, n

    
    
 
    
    
    
    
            
            