"""
This module contains the tests for timeconv function
"""
# Standard library imports


# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np
from math import gamma, fsum


# Local application imports
from myorbit.util.stumpff import calc_stumpff_exact, calc_stumpff_as_series

#
#  The Stumpff functions implemented by polyastro will be used as oracle for the tests
#
def poli_c2(psi):
    r"""Second Stumpff function.
    For positive arguments:
    .. math::
        c_2(\psi) = \frac{1 - \cos{\sqrt{\psi}}}{\psi}
    """
    eps = 1.0
    if psi > eps:
        res = (1 - np.cos(np.sqrt(psi))) / psi
    elif psi < -eps:
        res = (np.cosh(np.sqrt(-psi)) - 1) / (-psi)
    else:
        res = 1.0 / 2.0
        delta = (-psi) / gamma(2 + 2 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 2 + 1)

    return res



def poli_c3(psi):
    r"""Third Stumpff function.
    For positive arguments:
    .. math::
        c_3(\psi) = \frac{\sqrt{\psi} - \sin{\sqrt{\psi}}}{\sqrt{\psi^3}}
    """
    eps = 1.0
    if psi > eps:
        res = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi))
    elif psi < -eps:
        res = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / (-psi * np.sqrt(-psi))
    else:
        res = 1.0 / 6.0
        delta = (-psi) / gamma(2 + 3 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 3 + 1)

    return res


xs=np.linspace(-100,100,10000)

def calc_diffs(xs) :
    diff_c2_series=[]
    diff_c3_series=[]
    diff_c2_exact=[]
    diff_c3_exact=[]
    for x in xs:
        exp_c2 = poli_c2(x)
        exp_c3 = poli_c3(x)    
        _, c2_series, c3_series = calc_stumpff_as_series(x)
        _, c2_exact, c3_exact = calc_stumpff_exact(x)
        diff_c2_series.append(np.abs(c2_series-exp_c2))
        diff_c3_series.append(np.abs(c3_series-exp_c3))
        diff_c2_exact.append(np.abs(c2_exact-exp_c2))
        diff_c3_exact.append(np.abs(c3_exact-exp_c3))
    return fsum(diff_c2_series), fsum(diff_c3_series), fsum(diff_c2_exact), fsum(diff_c3_exact)


def test_small_values():
    xs=np.linspace(-0.1,0.1,10000)
    dc2_series, dc3_series, dc2_exact, dc3_exact = calc_diffs(xs)
    assert dc2_series == approx(0.0, abs=1.e-12)
    assert dc3_series == approx(0.0, abs=1.e-12)
    assert dc2_exact == approx(0.0, abs=1.e-10)
    assert dc3_exact == approx(0.0, abs=1.e-10)
    
def test_big_values():
    xs=np.linspace(0,1000,1000)
    dc2_series, dc3_series, dc2_exact, dc3_exact = calc_diffs(xs)
    assert dc2_series == approx(0.0, abs=1.e-4)
    assert dc3_series == approx(0.0, abs=1.e-4)
    assert dc2_exact == approx(0.0, abs=1.e-15)
    assert dc3_exact == approx(0.0, abs=1.e-15)
    
