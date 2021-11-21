""" This module provides the functionality to calculate ephemeris for two bodies problem
also in the case of perturbed methods. More advance pertubed methods will be handled  
in other module
"""

# Standard library imports
import logging
from math import isclose
from typing import ForwardRef

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
from myorbit.util.general import mu_Sun, calc_eccentricity_vector, angle_between_vectors
from myorbit.pert_cowels import calc_eph_by_cowells
from myorbit.two_body import calc_eph_planet

from myorbit.util.timeut import EQX_B1950, EQX_J2000
from myorbit.ephemeris_input import EphemrisInput
from myorbit.pert_enckes import calc_eph_by_enckes

from myorbit.two_body import calc_eph_twobody





from myorbit.util.constants import *

logger = logging.getLogger(__name__)


def calc_tp(M0, a, epoch):
    deltaT = TWOPI*np.sqrt(pow(a,3)/GM)*(1-M0/TWOPI)
    return deltaT + epoch

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
        T0_MJD = obj.tp_mjd-delta_days
        r0_xyz, rdot0_xyz, r0, h0_xyz, _ , f0 = solver.calc_rv(T0_MJD)    
        hs = []
        es = []
        for dt in range(2,delta_days*2,2):
            clock_mjd = T0_MJD + dt 
            try :
                r_xyz, rdot_xyz, h_xyz, f = calc_rv_from_r0v0(mu_Sun, r0_xyz, rdot0_xyz, dt, f0)
                hs.append(np.linalg.norm(h_xyz))
                es.append(np.linalg.norm(calc_eccentricity_vector(r_xyz, rdot_xyz,h_xyz)))
            except NoConvergenceError :
                print (f"===== Object {name} doest not converged at {clock_mjd} MJD")                              
                not_converged.append(name)
        if not all(isclose(h, hs[0], abs_tol=1e-12) for h in hs):
            msg = f'The angular momentum is NOT constant in the orbit'
            print (msg)
            logger.error(msg)
        if not all(isclose(ec, es[0], abs_tol=1e-12) for ec in es):
            msg = f'The eccentric vector  is NOT constant in the orbit'
            print (msg)
            logger.error(msg)
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

def test_comets_convergence(delta_days=50):
    df = dc.DF_COMETS
    #FILTERED_OBJS = ['C/1680 V1', 'C/1843 D1 (Great March comet)', 'C/1882 R1-A (Great September comet)', 'C/1882 R1-B (Great September comet)', 'C/1882 R1-C (Great September comet)', 'C/1882 R1-D (Great September comet)', 'C/1963 R1 (Pereyra)', 'C/1965 S1-A (Ikeya-Seki)', 'C/1965 S1-B (Ikeya-Seki)', 'C/1967 C1 (Seki)', 'C/1970 K1 (White-Ortiz-Bolelli)', 'C/2004 V13 (SWAN)', 'C/2011 W3 (Lovejoy)', 'C/2013 G5 (Catalina)', 'C/2020 U5 (PANSTARRS)']
    #FILTERED_OBJS=['C/1827 P1 (Pons)']
    FILTERED_OBJS=[]
    if len(FILTERED_OBJS) != 0:
        df = df[df.Name.isin(FILTERED_OBJS)]
    result = []
    df = df.sort_values('e', ascending=False)
    for idx, name in enumerate(df['Name']): 
        obj = dc.read_comet_elms_for(name,df)
        solver = KeplerianStateSolver.make(e=obj.e, a=obj.a, tp_mjd=obj.tp_mjd, q=obj.q, epoch=obj.epoch_mjd)
        T0_MJD = obj.tp_mjd-delta_days
        r0_xyz, rdot0_xyz, r0, h0_xyz, _ , f0 = solver.calc_rv(T0_MJD)            
        kep_nc = uni_nc = 0
        #print (f"Object {name} with e={obj.e}")
        for dt in range(2,delta_days*2,2):
            r1_xyz = rdot1_xyz = f1 = None
            try :
                r1_xyz, rdot1_xyz, r1, h1_xyz, _ , f1 = solver.calc_rv(T0_MJD+dt)    
            except NoConvergenceError :
                kep_nc += 1
            r2_xyz = rdot2_xyz = f2 = None
            try :                
                r2_xyz, rdot2_xyz, h_xyz, f2 = calc_rv_from_r0v0(mu_Sun, r0_xyz, rdot0_xyz, dt, f0)
            except NoConvergenceError :
                uni_nc += 1
                print (f"The noconvergence was with e: {obj.e}")
        if (kep_nc >0)  or (uni_nc > 0) :
            row = {}
            row['name'] = name
            row['e'] = obj.e
            row['kep_nc'] = kep_nc
            row['uni_nc'] = uni_nc
            result.append(row)            
    df_out = pd.DataFrame(result)
    if len(df_out) > 0:
        print (f'There are {len(df_out)} comets with convergence problems')
        df_out = df_out.sort_values(by=['uni_nc','kep_nc'],ascending=False)
        df_out.to_csv('convergence_problems.csv',index=False,header=True)
    else :
        print ("Undetected no-convergences")


def test_universal_kepler(delta_days=50):
    df = dc.DF_COMETS
    FILTERED_OBJS=[]
    #FILTERED_OBJS=['C/1933 D1 (Peltier)','C/1989 R1 (Helin-Roman)','C/2007 M5 (SOHO)','C/1988 M1 (SMM)','C/2008 C5 (SOHO)']    
    #FILTERED_OBJS=['C/2007 M5 (SOHO)']    
    # C/2000 O1 (Koehn)
    # This one has high nonconverence with 500 C/2000 O1 (Koehn)
    if len(FILTERED_OBJS) != 0:
        df = df[df.Name.isin(FILTERED_OBJS)]
    df = df.sort_values('e', ascending=False)
    result = []
    for idx, name in enumerate(df['Name']): 
        obj = dc.read_comet_elms_for(name,df)
        #print (name)
        solver = KeplerianStateSolver.make(e=obj.e, a=obj.a, tp_mjd=obj.tp_mjd, q=obj.q, epoch=obj.epoch_mjd)
        T0_MJD = obj.tp_mjd-delta_days
        r0_xyz, rdot0_xyz, r0, h0_xyz, _ , f0 = solver.calc_rv(T0_MJD)        
        r_failed = v_failed =  f_failed = nc_failed= 0 
        for dt in range(2,delta_days*2,2):
            try :
                r1_xyz, rdot1_xyz, r1, h1_xyz, _ , f1 = solver.calc_rv(T0_MJD+dt)    
                r2_xyz, rdot2_xyz, h2_xyz, f2 = calc_rv_from_r0v0(mu_Sun, r0_xyz, rdot0_xyz, dt, f0)
                e_xyz = calc_eccentricity_vector(r1_xyz, rdot1_xyz, h1_xyz)
                f3 = angle_between_vectors(e_xyz, r1_xyz)
                if not isclose(f1,f2,rel_tol=0, abs_tol=1e-03):
                    f_failed += 1
                    msg=f"name: {obj.name},  TWOPI - f univ: {TWOPI-f2} f Universal: {f2}  f Kepler: {f1} e:{obj.e}  f Excentricity: {f3}  f Excentricity: {TWOPI-f3}"
                    logger.error(msg)
                if not my_isclose(r1_xyz, r2_xyz, abs_tol=1e-03):
                    msg = f"name: {obj.name}, e: {obj.e}, diff_rxyz ={np.linalg.norm(r1_xyz- r2_xyz)}  diff_rdotxyz: {np.linalg.norm(rdot1_xyz- rdot2_xyz)}"
                    logger.error(msg)
                    r_failed += 1
                if not my_isclose (rdot1_xyz, rdot2_xyz, abs_tol=1e-03) :
                    v_failed += 1
            except NoConvergenceError :
                nc_failed += 1
        if (f_failed >0)  or (r_failed > 0) or (v_failed > 0) or (nc_failed > 0):
            row = {}
            row['name'] = name
            row['e'] = obj.e
            row['f_failed'] = f_failed
            row['r_failed'] = r_failed
            row['v_failed'] = v_failed
            row['nc_failed'] = nc_failed
            result.append(row)            
    df_out = pd.DataFrame(result)
    if len(df_out) > 0:
        print (f'There are {len(df_out)} comets with convergence problems')
        #df_out = df_out.sort_values(by=['uni_nc','kep_nc'],ascending=False)
        df_out.to_csv('kepler_universal.csv',index=False,header=True)
        print (df_out)
    else :
        print ("No problems detected")

def test_enckes():
    obj= dc.C_2003_M3_SOHO
    eph  = EphemrisInput(from_date="2001.03.01.0",
                        to_date = "2005.08.31.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    dfc = calc_eph_by_enckes(obj, eph)   


    
def test_comet(name, delta_days=50):
    obj = dc.read_comet_elms_for(name,dc.DF_COMETS)
    solver = KeplerianStateSolver.make(e=obj.e, a=obj.a, tp_mjd=obj.tp_mjd, q=obj.q, epoch=obj.epoch_mjd)
    T0_MJD = obj.tp_mjd-delta_days
    #print (f"Time interval considered: from:{mjd2str_date(T0_MJD-delta_days)} to {mjd2str_date(T0_MJD+delta_days)}")
    r0_xyz, rdot0_xyz, r0, h0_xyz, _ , f0 = solver.calc_rv(T0_MJD)  
    max_diff_r = 0
    for dt in range(2,delta_days*2,2):
        try :
            print (f"{mjd2str_date(T0_MJD+dt)}")
            r1_xyz, rdot1_xyz, r1, h1_xyz, _ , f1 = solver.calc_rv(T0_MJD+dt)    
            r2_xyz, rdot2_xyz, h2_xyz, f2 = calc_rv_from_r0v0(mu_Sun, r0_xyz, rdot0_xyz, dt, f0)
            if not isclose(f1,f2, rel_tol=0, abs_tol=1e-03):
                msg=f"{mjd2str_date(T0_MJD+dt)} f Uni:{f2}  f Kepler:{f1} TWOPI-f:{TWOPI-f1}"
                print (msg)
                logger.error(msg)
            if not my_isclose(r1_xyz, r2_xyz, abs_tol=1e-07):
                diff_rxyz = np.linalg.norm(r1_xyz- r2_xyz)
                if diff_rxyz > max_diff_r :
                    max_diff_r = diff_rxyz
                    print (f"Maximun distance at time:{mjd2str_date(T0_MJD+dt)}")
                msg = f"{mjd2str_date(T0_MJD+dt)}, diff_rxyz ={np.linalg.norm(r1_xyz- r2_xyz)}  diff_rdotxyz: {np.linalg.norm(rdot1_xyz- rdot2_xyz)}"
                print (msg)
                logger.error(msg)

        except NoConvergenceError :
            nc_failed += 1

def test_near_parabollic():
    obj=dc.C_2007_M5_SOHO
    eph  = EphemrisInput(from_date="2007.06.15.0",
                        to_date = "2007.07.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = EQX_J2000)

    df = calc_eph_twobody(obj, eph, force_orbit='near_parabolical')   
    #df = calc_eph_twobody(obj, eph)   
    print (df)
    

def change_reference_frame(heliocentric_orbs, name):
    orbs_from_obj = dict()
    # A new orbs object is created changing the frame of reference to the object (name of the object)
    # The object should be included in the helliocentric_orbs
    for body_name in filter(lambda x : x.lower()!=name.lower(), heliocentric_orbs.keys()):
        orbs_from_obj[body_name] = heliocentric_orbs[body_name] - heliocentric_orbs[name]    
    return orbs_from_obj

    
PLANET_NAMES= [x.lower() for x in GM_by_planet.keys()]    
    
def calc_orbits_heliocentric_data(eph, obj_names):
    """
    Computes the orbits of the planets, minor bodys and comets 
    
    Args:
        eph : EphemerisData
        planets : List of name of planets
        minor_bodys : List of names of minor bodys or orbital elements itself
        comets : List of names of comets bodys or orbital elements itself

    Returns :
        orbs : A dictionary where the key is the name of the body and value is a
               matrix of n,3 (n rows per 3 cols) with the heliocentric coordinates h_x, h_y, h_z
               and the index is the date of corresponding to the position.
        date_refs :  list of the dates where the heliocentric coordinates were calculated
        
    """    
    # orbs is a dictionary where the key is the name of the object (planet, asteroids or comet)
    # and the value is the dataframe with the ephemeris data.
    orbs = {}
    dfs = []
    for name in obj_names:
        if not isinstance(name, str):
            # Assumed that this is a BodyElms or CometElms 
            obj = name
            df  = calc_eph_by_cowells(obj,eph, include_osc=False)
            orbs[obj.name] = df
            dfs.append(df) 
            continue           
        if name.lower() in PLANET_NAMES:
            df = calc_eph_planet(name, eph)
            orbs[name] = df
            dfs.append(df)
        else :
            obj = dc.read_comet_elms_for(name,dc.DF_COMETS)        
            if obj is not None:
                df  = calc_eph_by_cowells(obj,eph, include_osc=False)
                orbs[name] = df
                dfs.append(df)
            else :
                obj = dc.read_body_elms_for(name,dc.DF_BODIES)
                if obj is not None:
                    df  = calc_eph_by_cowells(obj,eph, include_osc=False)
                    orbs[name] = df
                    dfs.append(df)
                else :
                    print (f"Object {name} not found")
    # Assumed that the ['date'] colum of each ephemeris are the same for every object so
    # we get the list of dates from the first object.
    first_key= list(orbs.keys())[0]
    date_refs = orbs[first_key]['date'].to_list()
    cols=['h_x','h_y','h_z']    
    for k, df in orbs.items():
        # For each object, only the ecliptic (heliocentric) coordinates are kept and
        # transformed to a matrix with shape (len(date_refs), 3)
        #    [[x1,y1,z1],
        #     [x2,y2,z2],
        #      ....
        #     [xn,yn,zn]]
        # for each key in the obr object, the value will be a nx3 matrix with the heliocentric coordinates
        orbs[k] = df[cols].to_numpy()     
    return orbs, dfs, date_refs
    
    
def calc_dangerous_asteroids(eph, n_objects=10):
    fname='/home/benito/PERSONAL/dangerous.csv'
    print (eph)
    df_out = pd.read_csv(fname,sep='|',names=['name', 'min_date','min_distance'])
    prev_len = len(dc.DF_BODIES)
    df = dc.DF_BODIES[~dc.DF_BODIES.Name.isin(df_out.name.values)]
    print (f"Filtered out {prev_len-len(df)} bodies")
    print ("Calculating Earth orbit data")
    orb_earth_from_Sun, *others = calc_orbits_heliocentric_data(eph, ['Earth'])
    with open(fname, 'at') as f:
        for idx, name in enumerate(df['Name']): 
            body = dc.read_body_elms_for(name,df)
            print (f"Processing {name}, Processed:{idx+1},  Remaining:{len(df)-idx}")            
            orb_obj_from_Sun, _, date_refs = calc_orbits_heliocentric_data(eph, [name])
            orb_obj_from_Earth = orb_obj_from_Sun[name] - orb_earth_from_Sun['Earth']
            distances_from_Earth = np.linalg.norm(orb_obj_from_Earth,axis=1)
            min_index = np.argmin(distances_from_Earth, axis=0)
            min_distance = distances_from_Earth[min_index]
            min_date = date_refs[min_index]
            f.write(f"{body.name}|{min_date}|{min_distance}\n")
            f.flush()
            if (idx > n_objects):
                break

if __name__ == "__main__":   
    #test_all_comets()
    #test_all_bodies()
    #test_almost_parabolical(50)
    #test_universal()
    #calc_comets_that_no_converge(20)
    #import logging.config
    #logging.config.fileConfig(CONFIG_INI, disable_existing_loggers=False)    
    #test_comets_convergence(5000)
    #test_universal_kepler(5000)
    #test_comet('C/2007 M5 (SOHO)',2500)
    #test_enckes()
    #test_near_parabollic()
    eph = EphemrisInput(from_date="2021.01.01.0",
                    to_date = "2060.12.01.0",
                    step_dd_hh_hhh = "05 00.0",
                    equinox_name = "J2000")
    calc_dangerous_asteroids(eph,n_objects=3000000)
    
    