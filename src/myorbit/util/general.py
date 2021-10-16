"""
This module contains functions related to time conversions
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html


from functools import partial
from itertools import tee
from functools import wraps
import datetime


# Third party imports
import numpy as np
import toolz as tz
from toolz import compose, pipe, valmap

from numpy import sin, cos, arccos
# Local application imports
# Should be empty in this case


# Product of grativational constant times and the solar mass AU^3*d^2
# This is equivalent to k^2 (Heliocentric Gaussian constant)
GM = 2.959122083e-4 
k_2 = GM
k_gauss = 0.01720209895
c_light = 173.14  # [AU/d]


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

AU_m = 149597870700
seconds_in_day = 3600*24

def to_AU_days(mu_m3s_2):
    return mu_m3s_2 * seconds_in_day*seconds_in_day/(AU_m*AU_m*AU_m)

# Gravitational parameters en AU/days
mu_by_name = valmap(to_AU_days,mu_m3s_2__by_name)

mu_Sun = mu_by_name["Sun"]


# Inverse of spped of light in days/AU
#INV_C = 0.00578 
INV_C = 0.0057755183


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def euclidean_distance (tup1, tup2) :
    return np.sqrt(np.square(tup1[0]-tup2[0])+np.square(tup1[0]-tup2[0]))


def pow(x,n):
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
    i = 0
    while start + i * step <= stop:
        yield start + i * step
        i += 1

def memoize(func):
    cache = func.cache = {}
    @wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func


def angular_distance(ra1,de1,ra2,de2):
    cos_d = sin(de1) * sin(de2) + cos(de1)*cos(de2)*cos(ra1-ra2)
    d = np.arccos(cos_d)
    return d    
 

def pr_rad(alpha):
    print (np.rad2deg(alpha))

def pr_radv(v):
    print(f'r: {v[0]}  lon: {np.rad2deg(v[1])}  lat: {np.rad2deg(v[2])}')


from functools import wraps
from time import time
def measure(func):
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
    """
    Generate a range of number but allows to force the inclusion of the start and the end
    the Cowells but it takes more time to be calculated.
    Even if the step is too big, the start and end can be forced

    Args:
        start :
        end : 
        step : 
        include_start :
        include_end

    Returns :
        A list with the interval
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
    