"""
This module contains functions related to time conversions
"""
# Standard library imports
from functools import wraps
from itertools import tee
from time import time

# Third party imports
import numpy as np
from numpy import cos, sin
from toolz import valmap

# Local application imports
from myorbit.util.constants import *

#https://en.wikipedia.org/wiki/Standard_gravitational_parameter
mu_m3s_2__by_name = {
    "Sun" : 1.32712440018e20,
    "Mercury" : 2.2032e13,
    "Venus" :3.24859e14,
    "Earth": 3.98604418e14,
    "Mars" : 4.282837e13,
    "Jupiter" : 1.26686534e17,
    "Saturn" : 3.7931187e16,
    "Uranus" : 5.793939e15,
    "Neptune" : 6.836529e15,
    "Pluto" : 8.71e11,
    "Ceres" : 6.26325e10,
    "Eris" : 1.108e12
}

PERTURBING_PLANETS = mu_m3s_2__by_name.keys() - ["Sun","Ceres","Eris"]

def to_AU_days(mu_m3s_2):
    return mu_m3s_2 * SECONDS_IN_DAY*SECONDS_IN_DAY/(AU_m*AU_m*AU_m)

# Gravitational parameters in AU/days
mu_by_name = valmap(to_AU_days,mu_m3s_2__by_name)

mu_Sun = mu_by_name["Sun"]

def pow(x,n):
    """Computes x^n 

    Parameters
    ----------
    x : float
        the value to power
    n : int
        The exponent 

    Returns
    -------
    int
        the value of x^n
    """
    if n ==0:
        return 1
    elif n==1:
        return x
    elif n==2:
        return x*x
    elif n==3: 
        return x*x*x
    else :
        return np.power(x,n)

def frange(start, stop, step):
    """Generates a list of integer numbers in a closed interval according to
    a step

    Parameters
    ----------
    start : int
        The start of the interval
    stop : int
        The end of the interval
    step : int
        the step

    Yields
    -------
    int
        the secuence of numbers
    """
    i = 0
    while start + i * step <= stop:
        yield start + i * step
        i += 1

def memoize(func):
    """Decorates a function adding caching funcionality, i.e, introduce a dictionary where
    the results are stored indexed by a key so the function is only called when
    the key requested is not in the dictionary

    Parameters
    ----------
    func : function
        Function to which the caching functionality is added

    Returns
    -------
    func
        The decorated function
    """
    cache = func.cache = {}
    @wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func


def angular_distance(ra1,de1,ra2,de2):
    """Computes the angular distance between two points from its RA,DEC coordinates

    Parameters
    ----------
    ra1 : float
        Right Ascenison of the first point [radians]
    de1 : float
        Declination of the first point [radians]
    ra2 : float
        Right Ascenison of the second point [radians]
    de2 : float
        Declination of the first point [radians]

    Returns
    -------
    float
        The angular distance [radians]
    """
    cos_d = sin(de1) * sin(de2) + cos(de1)*cos(de2)*cos(ra1-ra2)
    d = np.arccos(cos_d)
    return d    
 

def pr_rad(alpha):
    """Utility function to print in degrees an angle expressend in radians

    Parameters
    ----------
    alpha : float
        Angle [radians]
    """
    print (np.rad2deg(alpha))

def pr_radv(v):
    """Utility function to print in degrees a 3-vector that contains 
    a radial component (not an angle), a longitude (radians) and latitude (radians) 

    Parameters
    ----------
    v : np.array
        A 3-vector containing a radial component, a longitude component (radians) 
        and latitude component (radians)
    """
    print(f'r: {v[0]}  lon: {np.rad2deg(v[1])}  lat: {np.rad2deg(v[2])}')

def measure(func):
    """Decorates a function to measure its execution time

    Parameters
    ----------
    func : function
        The function whose execution time will be measured

    Returns
    -------
    decorator
        The function decorated
    """
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it
    


def my_range (start, stop, step, include_start = True, include_end = True):
    """Generate a range of number but allows to force the inclusion of the
    start and the end (used in EncKes) but it takes more time to be calculated.
    Even if the step is too big, the start and end can be forced

    Parameters
    ----------
    start : int
        The inital index of the range
    stop : int
        The final index of the range
    step : int
        The step
    include_start : bool, optional
        Indicates whether the initialindex must be included in the result or not[description], by default True
    include_end : bool, optional
        Indicates whether the final index must be included in the result or not, by default True

    Returns
    -------
    list
        The list of indexes that meets the criteria
    """

    result = []
    i = 0
    while start + i * step <= stop:
        result.append(start + i * step)
        i += 1
    if include_end :         
        if result[-1] != stop :
            result.append(stop)
    else :
        if result[-1] == stop :
            result = result[:-1]
    if not include_start :         
        result = result[1:]
    return result
    
if __name__ == "__main__" :
   None