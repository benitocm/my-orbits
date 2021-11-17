"""This module contains the class for entering the input data for ephemeris calculation
"""

# Standard library imports
import logging

# Third party imports
from toolz import pipe

# Local application imports
from myorbit.util.timeut import epochformat2jd, jd2mjd, T, jd2str_date, mjd2jd
from myorbit.util.constants import *

logger = logging.getLogger(__name__)


def step_from_str(step_str):
    toks = step_str.split(' ')
    if len(toks) ==1 :
        return float(toks[0])
    else :        
        return float(toks[0])+float(toks[1])/24.0

class EphemrisInput:
    def __init__(self, from_date="", to_date="", step_dd_hh_hhh="", equinox_name=""):
        self.from_date = from_date
        self.to_date = to_date
        self.from_mjd = pipe(epochformat2jd(from_date), jd2mjd)
        self.to_mjd = pipe(epochformat2jd(to_date), jd2mjd)
        self.eqx_name = equinox_name
        self.T_eqx = T(equinox_name)
        self.step = step_from_str(step_dd_hh_hhh)
    
    @classmethod
    def from_mjds(cls, from_mjd, to_mjd, step_dd_hh_hhh="", equinox_name=""):    
        obj = cls.__new__(cls)  # Does not call __init__
        super(EphemrisInput, obj).__init__()  # Don't forget to call any polymorphic base class initializers
        obj.from_date = jd2str_date (mjd2jd(from_mjd))
        obj.to_date= jd2str_date(mjd2jd(to_mjd))
        obj.from_mjd = from_mjd
        obj.to_mjd = to_mjd
        obj.eqx_name = equinox_name
        obj.T_eqx = T(equinox_name)
        obj.step = step_from_str(step_dd_hh_hhh)
        return obj


    def __str__(self):
        s = []
        s.append(f'     Equinox name: {self.eqx_name}')
        s.append(f'            T eq0: {self.T_eqx}')
        s.append(f'        From date: {self.from_date}')
        s.append(f'          To date: {self.to_date}')
        s.append(f'         From MJD: {self.from_mjd}')
        s.append(f'           To MJD: {self.to_mjd}')
        s.append(f'      Step (days): {self.step}')
        return '\n'.join(s)    

