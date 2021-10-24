"""
This module contains the tests for timeconv function
"""
# Standard library imports
# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np
from numpy.random import default_rng

# Local application imports
from myorbit.util.timeut import *

def test_myfix():
    assert my_fix(1.1) == 1
    assert my_fix(-1.1) == -1
    assert my_fix(1.99) == 1
    assert my_fix(-1.99) == -1


def test_myfrac():
    assert my_frac(1.9) == approx(.9)
    assert my_frac(1.1) == approx(.1)
    assert my_frac(-1.9) == approx(.9)
    assert my_frac(-1.1) == approx(.1)
