""" This module provides the functionality to calculate ephemeris for two bodies problem
also in the case of perturbed methods. More advance pertubed methods will be handled  
in other module
"""

# Standard library imports
import logging
from math import isclose

# Third party imports
import pandas as pd
import numpy as np
from numpy.linalg import norm
from toolz import pipe

# Local application imports
from myorbit.util.general import  my_range,  NoConvergenceError
import myorbit.data_catalog as dc
from myorbit.util.timeut import mjd2str_date
from myorbit.planets import g_xyz_equat_sun_j2000
from myorbit.orbits.keplerian import KeplerianStateSolver, ParabolicalStateSolver, EllipticalStateSolver
from myorbit.orbits.ellipitical import calc_rv_for_elliptic_orbit, calc_M

from myorbit.util.constants import *

logger = logging.getLogger(__name__)


def calc_tp(M0, a, epoch):
    deltaT = TWOPI*np.sqrt(pow(a,3)/GM)*(1-M0/TWOPI)
    return deltaT + epoch


    """The orbit of all comets is studied around the perihelion [-days, +days]
    """

def calc_comets_that_no_converge(delta_days):
    """The orbit of all comets is studied around the perihelion [-days, +days]

    Parameters
    ----------
    delta_days : int
        [description]
    """
    df = dc.DF_COMETS
    not_converged=[]
    for idx, name in enumerate(df['Name']): 
        obj = dc.read_comet_elms_for(name,df)
        msg = f'Testing Object: {obj.name}'
        print (msg)
        logger.info(msg)
        if hasattr(obj,'M0') :
            M_at_epoch = obj.M0
        else :
            M_at_epoch = None
        # from 20 days before perihelion passage to 20 days after 20 days perihelion passage
        solver = KeplerianStateSolver.make(e=obj.e, a=obj.a, tp_mjd=obj.tp_mjd, q=obj.q, epoch=obj.epoch_mjd, M_at_epoch=M_at_epoch)
        hs = []
        try :
            for clock_mjd in my_range(obj.tp_mjd-delta_days, obj.tp_mjd+delta_days, 2):        
                r_xyz, rdot_xyz, r, h = solver.calc_rv(clock_mjd)
                hs.append(h)
            if not all(isclose(h, hs[0], abs_tol=1e-12) for h in hs):
                msg = f'The angular momentum is NOT constant in the orbit'
                print (msg)
                logger.error(msg)
        except NoConvergenceError :
            print (f"===== Object {name} doest not converged at {clock_mjd} MJD")                              
            not_converged.append(name)
    print (not_converged)
    
def test_all_bodies(delta_days):
    df = dc.DF_BODIES
    not_converged=[]
    for idx, name in enumerate(df['Name']): 
        body = dc.read_body_elms_for(name,df)
        msg = f'Testing Object: {body.name}'
        solver = KeplerianStateSolver.make(e=body.e, a=body.a, epoch=body.epoch_mjd, M_at_epoch=body.M0)    
        tp = calc_tp(body.M0, body.a, body.epoch_mjd) 
        hs = []
        try :
            for clock_mjd in my_range(tp-delta_days, tp+delta_days, 2):        
                r_xyz, rdot_xyz, r, h = solver.calc_rv(clock_mjd)
                hs.append(h)
            if not all(isclose(h, hs[0], abs_tol=1e-12) for h in hs):
                msg = f'The angular momentum is NOT constant in the orbit'
                print (msg)
                logger.error(msg)
        except NoConvergenceError :
            print (f"===========> NOT converged for object {name}")
            not_converged.append(name)    
        if idx % 1000 == 0 :
            print (f"================================================>> {idx}")
    print (not_converged)

def test_almost_parabolical(delta_days):
    df = dc.DF_COMETS
    not_converged=[]
    names = ['C/1680 V1', 'C/1843 D1 (Great March comet)', 'C/1882 R1-A (Great September comet)', 'C/1882 R1-B (Great September comet)', 'C/1882 R1-C (Great September comet)', 'C/1882 R1-D (Great September comet)', 'C/1963 R1 (Pereyra)', 'C/1965 S1-A (Ikeya-Seki)', 'C/1965 S1-B (Ikeya-Seki)', 'C/1967 C1 (Seki)', 'C/1970 K1 (White-Ortiz-Bolelli)', 'C/2004 V13 (SWAN)', 'C/2011 W3 (Lovejoy)', 'C/2013 G5 (Catalina)', 'C/2020 U5 (PANSTARRS)']
    #names = ['C/2020 U5 (PANSTARRS)']
    df = df[df.Name.isin(names)]
    for idx, name in enumerate(df['Name']): 
        if name not in names :
            continue
        obj = dc.read_comet_elms_for(name,df)
        msg = f'Testing Object: {obj.name} with Tp:{mjd2str_date(obj.tp_mjd)}'
        print (msg)
        logger.info(msg)
        if hasattr(obj,'M0') :
            M_at_epoch = obj.M0
        else :
            M_at_epoch = None
        # from 20 days before perihelion passage to 20 days after 20 days perihelion passage
        #solver = ParabolicalStateSolver(obj.tp_mjd, obj.q, obj.e)
        solver = EllipticalStateSolver(q=obj.q, a=obj.a, e=obj.e, tp_mjd=obj.tp_mjd, epoch_mjd=obj.epoch_mjd)
        hs = []
        for clock_mjd in my_range(obj.tp_mjd-delta_days, obj.tp_mjd+delta_days, 2):               
            r_xyz, rdot_xyz, r, h_xyz, *others = solver.calc_rv(clock_mjd)
            hs.append(h_xyz)
            print(mjd2str_date(clock_mjd))    

        if not all(np.allclose(h_xyz, hs[0], atol=1e-12) for h_xyz in hs):
            msg = f'The angular momentum is NOT constant in the orbit'
            print (msg)
            logger.error(msg)
        print (not_converged)




if __name__ == "__main__":
    #test_all_comets()
    #test_all_bodies()
    test_almost_parabolical(50)
    