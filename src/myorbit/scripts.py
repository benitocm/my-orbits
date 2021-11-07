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
from myorbit.util.general import  my_range,  NoConvergenceError, my_isclose
import myorbit.data_catalog as dc
from myorbit.util.timeut import mjd2str_date
from myorbit.planets import g_xyz_equat_sun_j2000
from myorbit.kepler.keplerian import KeplerianStateSolver, ParabolicalStateSolver, EllipticalStateSolver
from myorbit.kepler.ellipitical import calc_rv_for_elliptic_orbit, calc_M
from myorbit.lagrange.lagrange_coeff import calc_rv_from_r0v0
from myorbit.util.general import mu_Sun

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

def test_universal():

    # Elliptical comet
    #C2012_CH17 = read_comet_elms_for("C/2012 CH17 (MOSS)", DF_COMETS)   

    # Hyperbolic comet
    #C_2020_J1_SONEAR = read_comet_elms_for("C/2020 J1 (SONEAR)", DF_COMETS) 

    # Parabolic comet:
    #C_2018_F3_Johnson = read_comet_elms_for("C/2018 F3 (Johnson)", DF_COMETS) 

    # Near parabolic comet:
    #C_2011_W3_Lovejoy = read_comet_elms_for("C/2011 W3 (Lovejoy)", DF_COMETS) 

    #obj = dc.HALLEY_J2000    
    #OBJS=[dc.C_2020_J1_SONEAR, dc.C2012_CH17, dc.C_2018_F3_Johnson, dc.C_2011_W3_Lovejoy]
    #OBJS=[dc.C2012_CH17, dc.C_2011_W3_Lovejoy]
    OBJS=[dc.C2012_CH17]
    delta_days = 100
    for obj in OBJS:     
        #print (f"Testing {obj.name} ")
        solver = KeplerianStateSolver.make(e=obj.e, a=obj.a, tp_mjd=obj.tp_mjd, q=obj.q, epoch=obj.epoch_mjd)
        T0_MJD = obj.tp_mjd-delta_days
        r0_xyz, rdot0_xyz, r0, h0_xyz, *others = solver.calc_rv(T0_MJD)    
        r_failed = v_failed = 0
        for dt in range(2,delta_days*2,2):
            #print (f'Time: {T0_MJD+dt}  {mjd2str_date(T0_MJD+dt)}')
            r1_xyz, rdot1_xyz, *other = calc_rv_from_r0v0(mu_Sun, r0_xyz, rdot0_xyz, dt)
            print (f'State Keplerian:  r_xyz:{r1_xyz}, rdot_xyz:{rdot1_xyz}')
            print (f'State Universal:  r_xyz:{r2_xyz}, rdot_xyz:{rdot2_xyz}')
            if not my_isclose(r1_xyz, r2_xyz, abs_tol=1e-08):
                r_failed += 1
            if not my_isclose (rdot1_xyz, rdot2_xyz) :
                v_failed += 1
        print (f'>>>>>>>>>>> Object {obj.name} has r_failed:{r_failed} v_failed:{v_failed}')
    



if __name__ == "__main__":
    #test_all_comets()
    #test_all_bodies()
    #test_almost_parabolical(50)
    test_universal()
    