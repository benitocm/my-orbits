"""
This module contains the tests for timeconv function
"""
# Standard library imports


# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np


# Local application imports
from myorbit.util.general import my_range



def my_range2 (start, stop, step, include_start = True, include_end = True):
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


def test_my_range1():
    # By default the both ends are enforced to be included
    interval = my_range(1,10,2)
    assert interval == [1,3,5,7,9,10]
    # Even if the step is to long
    interval = my_range(1,10,20)
    assert interval == [1,10]









    
