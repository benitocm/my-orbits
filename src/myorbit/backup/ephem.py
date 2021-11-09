"""
This module contains functions related to orbit calculations
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

from datetime import datetime
from functools import partial
from collections import namedtuple
from math import isclose
import logging

# Third party imports
import pandas as pd
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, arctan, sqrt,cosh,sinh,deg2rad,rad2deg, sqrt
from numpy.linalg import multi_dot, norm
from toolz import pipe, compose, first

# Local application imports

import myorbit.planets as pl
from myorbit import coord as co
from myorbit.util.general import frange
from myorbit.util.timeut import reduce_rad
from myorbit.coord import mtx_eclip_prec, mtx_equatFeclip, format_time, format_dg
from myorbit.util.timeut import T_J2000,  CENTURY, JD_J2000, EQX_B1950, EQX_J2000, dg2h, h2hms, dg2dgms, T_given_mjd, mjd2jd, jd2str_date
from myorbit.orbits.keplerian import KeplerianOrbit, _parabolic_orbit, _hyperpolic_orbit, _elliptic_orbit, _next_H, _next_E
from myorbit.orbits.orbutil import process_solution, do_integration, my_f, calc_osculating_orb_elmts
from myorbit.planets import h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000, g_rlb_eclip_sun_eqxdate, g_xyz_equat_sun_j2000
import myorbit.data_catalog as dc
from myorbit.orbits.ephemeris_input import EphemrisInput
import myorbit.util.timeut as ut

from myorbit.util.constants import *

logger = logging.getLogger(__name__)

def calc_M(t:float, tp:float, a:float) -> float :
    """ 
    Computes the mean anomaly as a function a t, t0 and a, i.e., not depending on the
    period of the orbit=1)

    Args:
        t : time of the computation 
        t0: Time of perihelion passage 
        a: semi-major axis in AU

    Returns :
        The mean anomaly in radians
    """
    M = sqrt(GM)*(t-tp)/np.float_power(a,1.5)
    return reduce_rad(M, to_positive=True)

def calc_semimajor_axis(e,q):
    """ 
    Computes the semi-major axis 

    Args:
        e : Eccentricity of the orbit
        q : Perihelion distance q in AU
        
    Returns :
        The semi-major axis in AU
    """
    return q/(1.0-e)

def h_xyz_eclip_keplerian_orbit (q, e, mtx_PQR, tp, t) :
    """ 
    Computes position (r) and velocity (v) vectors for keplerian orbits
    depending the eccentricy and mean_anomlay to choose which type of conic
    use.

    Args:
        q: Perihelion distance q in AU
        e: Eccentricity of the orbit
        mtx_U : Matrix to change from orbital plane to eclictic plane
        tp : Time reference at which the object passed through perihelion
        t : time reference where the r, v vector will be calculated

    Returns :
        Position (r): It is a np.array of 3 elements with the cartesian coordinates w.r.t. the ecliptic
        Velocity (v): It is a np.array of 3 elements with the cartesian coordinates w.r.t. the ecliptic
    """
    M0 = 0.1 # radians
    epsilon = 0.1
    a = calc_semimajor_axis(e,q)    
    M = pipe( calc_M(t, tp, a), reduce_rad)         
    if (M < M0) and (np.abs(1.0-e)< epsilon) :
        r,v = _parabolic_orbit(tp, q, e, t,50)
        #r,v = elliptic_orbit(next_E, M, a, e)
    elif (e < 1.0) :
        r,v = _elliptic_orbit(_next_E, M, a, e)
    else :
        r,v =  _hyperpolic_orbit (tp, _next_H, a, e, t)
    # The r and v are transformed from the orbital plane to the ecliptic coordinate system
    # althouth they keep being cartesian coordinates.
    return mtx_PQR.dot(r),  mtx_PQR.dot(v)

def calc_eph_minor_body_perturbed (body, eph , include_osc=False):
    """
    Computes the ephemeris for a minor body 

    Args:
        body : The orbital elements of the body (CometElms)
        eph : The ephemeris data
        include_osc : Flag to include or not the osculating 

    Returns :
        A dataframe with the epehemris calculated.
    """

    # This matrix just depends on the desired equinox to calculate the obliquity
    # to pass from ecliptic coordinate system to equatorial
    MTX_equatFeclip = co.mtx_equatFeclip(eph.T_eqx)

    T_J2000 = 0.0    
    # This is to precess from J2000 to ephemeris equinox (normally this matrix will be identity matrix)
    MTX_J2000_Teqx = co.mtx_eclip_prec(T_J2000,eph.T_eqx)

    # The PQR mtx (from orbital plane to eclipt) is preccesed from equinox of body object to the desired equinox 
    MTX_J2000_PQR = co.mtx_eclip_prec(body.T_eqx0, T_J2000).dot(body.mtx_PQR)
    
    # The initial conditions for doing the integration is calculated, i.e.,
    # the r,v of the body at its epoch (in the example of Ceres, the epoch of 
    # book that epoch  is 1983/09/23.00)
    # The integration is done in the ecliptic plane and precessed in to J2000
    # so the solution will be also ecliptic and precessed.


    initial_mjd = body.epoch_mjd  
    M = body.calc_M(initial_mjd)
    M0 = 0.1  
    if ((M < M0) and ((1.0-body.e) < 0.1)) or isclose(body.e, 1.0,  abs_tol=1e-08) :
        logger.warning(f'Doing parabolic orbit for e: {body.e}')
        xyz0, vxyz0 = _parabolic_orbit(body.tp_mjd, body.q, body.e, initial_mjd, 50)
    elif body.e < 1.0 :
        logger.warning(f'Doing elliptic orbit for e: {body.e}')
        xyz0, vxyz0 = _elliptic_orbit(_next_E, M, body.a, body.e)
    else :
        logger.warning(f'Doing hyperbolic orbit for e: {body.e}')
        xyz0, vxyz0 =  _hyperpolic_orbit (body.tp_mjd, _next_H, np.abs(body.a), body.e, initial_mjd)

    y0 = np.concatenate((MTX_J2000_PQR.dot(xyz0), MTX_J2000_PQR.dot(vxyz0)))  
    
    # The integration is done in the ecliptic plane. First, we propagete the state vector
    # from the epoch of the body (when the data are fresh) up to the start of the ephemeris
    # in that interval we dont request the solution. 
    # Second we integrate in the ephemeris interval and asking the solution every step 
    # (e.g. 2 days) this is t.sol time samples
    # In case the epoch of the objet is at the future, the integration is done backward
    # in time (during the same interval but in reverse mode)

    if eph.from_mjd < initial_mjd < eph.to_mjd :
        # We need to do 2 integrations
        # First one backwards
        t_sol = list(reversed(list(frange(eph.from_mjd, initial_mjd, eph.step))))
        sol_1 = do_integration(my_f, y0, initial_mjd, t_sol[-1], t_sol)
        # Second one forwards
        t_sol = list(frange(initial_mjd, eph.to_mjd, eph.step))
        sol_2 = do_integration(my_f, y0, initial_mjd, t_sol[-1], t_sol)

        SOL_T = np.concatenate((sol_1.t, sol_2.t))
        SOL_Y = np.concatenate((sol_1.y, sol_2.y), axis=1)
    else :
        t_sol = list(frange(eph.from_mjd, eph.to_mjd, eph.step))
        if eph.to_mjd < initial_mjd :
            # If the epoch is in the future, we need to integrate backwards, i.e.
            # propagatin the state vector from the future to the past.
            t_sol = list(reversed(t_sol))
        sol = do_integration(my_f, y0, initial_mjd, t_sol[-1], t_sol)       
        SOL_T = sol.t
        SOL_Y = sol.y


    """
    t_sol = list(ut.frange(eph.from_mjd, eph.to_mjd, eph.step))
    if eph.from_mjd < body_elms.epoch_mjd :
        # If the epoch is in the future, we need to integrate backwards, i.e.
        # propagatin the state vector from the future to the past.
        t_sol = list(reversed(t_sol))
    sol = ob.do_integration(ob.my_f, y0, initial_mjd, t_sol[-1], t_sol)       
    """

    cols = ['date','Sun(dg)','h_l','h_b','h_r','ra','dec','r[AU]','h_x','h_y','h_z','t_mjd']
    oscul_keys = []
    rows = []
    for idx, clock_mjd in enumerate(SOL_T) :  
        row = {}
        clock_jd = mjd2jd(clock_mjd)
        row['date'] = jd2str_date(clock_jd)
        row['t_mjd'] = clock_mjd
        # Because the integration is done in the ecliptic coordinate system and precessed
        # so the solution will be already ecliptic so we just need to precess it to the 
        # desired equinox before transforming it to the equatorial
        h_xyz, h_vxyz = (SOL_Y[:,idx][:3], SOL_Y[:,idx][3:6])

        h_xyz = MTX_J2000_Teqx.dot(h_xyz)
        h_vxyz = MTX_J2000_Teqx.dot(h_vxyz)

        oscul_elms = calc_osculating_orb_elmts(h_xyz, h_vxyz, clock_mjd, eph.eqx_name)
        row.update(oscul_elms.as_dict())
        if len(oscul_keys) == 0 :
           oscul_keys = list(oscul_elms.as_dict().keys())


        h_xyz_equat_body = MTX_equatFeclip.dot(h_xyz)
        h_vxyz_equat_body = MTX_equatFeclip.dot(h_vxyz)

        # This is just to obtain the geo ecliptic longitud of the Sun and include in the
        # dataframe. Becasue the we have the sun position at equinox of the date,
        # we need to preccess it (in ecplitpical form) to J2000
        T = (clock_jd - JD_J2000)/CENTURY
        g_rlb_eclipt_T = pipe (g_rlb_eclip_sun_eqxdate(clock_jd, tofk5=True) , 
                               co.cartesianFpolar, 
                               mtx_eclip_prec(T,T_J2000).dot, 
                               co.polarFcartesian)
        row['Sun(dg)'] = f"{rad2deg(g_rlb_eclipt_T[1]):03.1f}"
        
        g_xyz_equat_sun = g_xyz_equat_sun_j2000(clock_jd)

        #We need to log the heliocentirc ecliptic coordinates in polar format 
        rlb = co.polarFcartesian(h_xyz)
        row['h_l'] = f"{rad2deg(rlb[1]):03.1f}"
        row['h_b'] = f"{rad2deg(rlb[2]):03.1f}"
        row['h_r'] = f"{rlb[0]:03.4f}"

        row['h_x'] = h_xyz[0]
        row['h_y'] = h_xyz[1]
        row['h_z'] = h_xyz[2]


        # Doing the sum of the vectors in the equaotiral planes works better for me.
        g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
        r_AU = norm(g_xyz_equat_body) 

        # The object will be closer because while the light traves to the earth
        # the object is moving. The correction for the ligth is done using
        # the aproach described in Astronomy on my computer
        # In Meeus, the methods implies to run two orbits keplerian. 
        # I prefer the first method because with perturbation we cannot execute
        # the second round needed.
        g_xyz_equat_body -= ut.INV_C*norm(g_xyz_equat_body)*h_vxyz_equat_body

        g_rlb_equat_body = co.polarFcartesian(g_xyz_equat_body)        

        row['ra'] = format_time(pipe(g_rlb_equat_body[1], rad2deg,  dg2h, h2hms))
        row['dec'] = format_dg(*pipe(g_rlb_equat_body[2], rad2deg, dg2dgms))
        row['r[AU]'] = r_AU
        rows.append(row)

    df = pd.DataFrame(rows)
    if include_osc :
        cols += oscul_keys     
    return df[cols].sort_values(by='t_mjd')


def calc_eph_planet(name, eph):
    """
    Computes the ephemris for a comet

    Args:
        body : The orbital elements of the body (CometElms)
        eph : The ephemeris data

    Returns :
        A dataframe with the epehemris calculated.
    """
    rows = []
    for clock_mjd in frange(eph.from_mjd, eph.to_mjd, eph.step):        
        row = {}
        clock_jd = mjd2jd(clock_mjd)
        row['date'] = jd2str_date(clock_jd)
        row['t_mjd'] = clock_mjd

        T = (clock_mjd - MDJ_J2000)/CENTURY    
        # desired equinox is J2000, so T_desired is 0
        T_desired = (JD_J2000 - JD_J2000)/CENTURY
        mtx_prec = co.mtx_eclip_prec(T, T_desired)

        if name == 'Pluto':
            h_xyz_eclipt = h_xyz_eclip_pluto_j2000(clock_jd)            
            g_rlb_equat = g_rlb_equat_pluto_j2000(clock_jd)
        else :
            # Planetary position (ecliptic and equinox of J2000)
            h_xyz_eclipt = mtx_prec.dot(h_xyz_eclip_eqxdate(name, clock_jd))
            g_rlb_equat = g_rlb_equat_planet_J2000(name,clock_jd)
        
        row['h_x'] = h_xyz_eclipt[0]
        row['h_y'] = h_xyz_eclipt[1]
        row['h_z'] = h_xyz_eclipt[2]

        row['ra'] = format_time(pipe(g_rlb_equat[1], rad2deg,  dg2h, h2hms))
        row['dec'] = format_dg(*pipe(g_rlb_equat[2], rad2deg, dg2dgms))
        row['r [AU]'] = norm(h_xyz_eclipt)
 
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(by='t_mjd')


def calc_eph_comet(body, eph):
    """
    Computes the ephemris for a comet

    Args:
        body : The orbital elements of the body (CometElms)
        eph : The ephemeris data

    Returns :
        A dataframe with the epehemris calculated.
    """

    # Normally, the equinox of the data of the body will be J2000  and the equinox of the 
    # ephemeris will also be J2000, so the precesion matrix will be the identity matrix
    # Just in the case of book of Personal Astronomy with your computer pag 81 is used   
    MTX_Teqx_PQR = mtx_eclip_prec(body.T_eqx0, eph.T_eqx).dot(body.mtx_PQR)  

    # Transform from ecliptic to equatorial just depend on desired equinox
    MTX_equatFecli = mtx_equatFeclip(eph.T_eqx)

    cols = ['date','Sun(dg)','h_l','h_b','h_r','ra','dec','r[AU]','h_x','h_y','h_z','t_mjd']

    rows = []
    for clock_mjd in frange(eph.from_mjd, eph.to_mjd, eph.step):        
        row = {}
        clock_jd = mjd2jd(clock_mjd)
        row['date'] = jd2str_date(clock_jd)
        row['t_mjd'] = clock_mjd

        # This is just to obtain the geo ecliptic longitud of the Sun and include in the
        # dataframe. Becasue the we have the sun position at equinox of the date,
        # we need to preccess it (in ecplitpical form) to J2000
        T = (clock_jd - JD_J2000)/CENTURY
        g_rlb_eclip = g_rlb_eclip_sun_eqxdate(clock_jd, tofk5=True) 
        g_rlb_eclipt_T = pipe (g_rlb_eclip, co.cartesianFpolar, mtx_eclip_prec(T,T_J2000).dot, co.polarFcartesian)
        row['Sun(dg)'] = f"{rad2deg(g_rlb_eclipt_T[1]):03.1f}"


        g_xyz_equat_sun = g_xyz_equat_sun_j2000(clock_jd)

        # Beause MTX_Teqx_PQR is used, the xyz of the body will be preccesed from the T_eq0
        # to desired equinox T_eqx, i.e., these are helicentric eclipt coordinates of the
        # object precesed to the correct equinox.
        h_xyz, h_vxyz = h_xyz_eclip_keplerian_orbit (body.q, body.e, MTX_Teqx_PQR, body.tp_jd, clock_jd)      

        #We need to log the heliocentirc ecliptic coordinates in polar format 
        rlb = co.polarFcartesian(h_xyz)
        row['h_l'] = f"{rad2deg(rlb[1]):03.1f}"
        row['h_b'] = f"{rad2deg(rlb[2]):03.1f}"
        row['h_r'] = f"{rlb[0]:03.4f}"

        row['h_x'] = h_xyz[0]
        row['h_y'] = h_xyz[1]
        row['h_z'] = h_xyz[2]

        h_xyz_equat_body = MTX_equatFecli.dot(h_xyz)
        g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
        r_AU = norm(g_xyz_equat_body) 

        # The object will be closer because while the light traves to the earth
        # the object is moving. The correction for the ligth is done using
        # the aproach described in Astronomy on my computer
        # In Meeus, the methods implies to run two orbits keplerian. 
        # I prefer the first method because with perturbation we cannot execute
        # the second round needed.
        g_xyz_equat_body -= ut.INV_C*norm(g_xyz_equat_body)*MTX_equatFecli.dot(h_vxyz)

        # Commented lines for the Meeus method pg 224
        #tau = np.linalg.norm(g_xyz_equat_body)*ut.INV_C
        #xyz, _ = h_xyz_eclip_keplerian_orbit (body.q, body.e, MTX_Teqx_PQR, body.tp_jd, clock_jd-tau)
        #h_xyz_equat_body = MTX_equatFecli.dot(xyz)
        #g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body

        g_rlb_equat_body = co.polarFcartesian(g_xyz_equat_body)        

        row['ra'] = format_time(pipe(g_rlb_equat_body[1], rad2deg,  dg2h, h2hms))
        row['dec'] = format_dg(*pipe(g_rlb_equat_body[2],rad2deg, dg2dgms))
        row['r[AU]'] = r_AU

        rows.append(row)
    df = pd.DataFrame(rows)
    return df[cols].sort_values(by='t_mjd')
    
"""
def h_xyz_eclip_body_j2000(jd):

    # Orbital elemetes

    Omega = deg2rad(58.14397)
    i = deg2rad(162.23932)
    omega = deg2rad(111.84658)
    #Tp_jd = tc.datefd2jd(1990,10,28.54502)
    e = 0.9672725
    q = 0.5870992
    a = q / (1-e)
    Tp_jd = tc.datefd2jd(1986,2,9.43867)

    # to pass from orbital plane to the ecliptic
    mtx_PQR = co.mtx_gauss_vectors(Omega, i, omega)
    gxyz_equat_sun = g_xyz_equat_sun_j2000(jd)    

    M = calc_M(jd, Tp_jd, a)
    xyz, _ = elliptic_orbit(next_E, M, a, e)
    hxyz_eclip_body = mtx_PQR.dot(xyz)
    return Coord(polarFcartesian(hxyz_eclip_body),'',co.ECLIP_TYPE)
"""

def g_rlb_equat_body_j2000(jd, body):
   
    T_J2000 = 0.0 # desired equinox

    # The matrix will include the precesion from equinox of the date of the body
    mtx_equat_PQR =  multi_dot([mtx_equatFeclip(T_J2000),
                               co.mtx_eclip_prec(body.T_eqx0,T_J2000),
                               co.mtx_gauss_vectors(body.Node, body.i, body.w)                                                          
                               ])

	# The equatorial xyz of the sun referred to equinox J2000 for the 
	# specific moment
    g_xyz_equat_sun = g_xyz_equat_sun_j2000(jd)    
    
    # Fist calculation of the position of the body is done
    M = calc_M(jd, body.tp_jd, body.a)
    M = reduce_rad(M,True)
    xyz, _ = elliptic_orbit(next_E, M, body.a, body.e)
	# xyz are cartesians in the orbital plane (perifocal system) so we need to transform 
	# to equatorial
    		
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    tau = np.linalg.norm(g_xyz_equat_body)*ut.INV_C

    # Second try using tau
    #M = ob.calc_M(jd-tau, body.tp_jd, body.a)
    M = calc_M(jd-tau, body.tp_jd, body.a)
    M = reduce_rad(M,True)
    xyz, _ = elliptic_orbit(next_E, M, body.a, body.e)
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)    
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    return polarFcartesian(g_xyz_equat_body)

"""	
def g_rlb_equat_body_j2000(jd, body, mtx_equat_PQR):

    Computes the geocentric equatorial coordinates of a body for J2000 at specific time 

    Args:
        jd : Point in time specified as Julian Day
        body: Orbital elements of the body (BodyElms)
        mtx_equat_PQR : The matrix to pass from orbital plane to ecplito and to equatorial. This is case
                        the matrix is used other calculations
        

    Returns :
        a numpy vector with the r,l,b attributes in radians


   
    T_J2000 = 0.0 # desired equinox

    # The matrix will include the precesion from equinox of the date of the body

	# The equatorial xyz of the sun referred to equinox J2000 for the 
	# specific moment
    g_xyz_equat_sun = g_xyz_equat_sun_j2000(jd)
    #Fist calculation of the position of the body is done

    M = calc_M(jd, body.tp_jd, body.a)
    M = reduce_rad (M,True)
    xyz, _ = elliptic_orbit(next_E, M, body.a, body.e)
	# xyz are cartesians in the orbital plane (perifocal system) so we need to transform 
	# to equatorial (cartesians) 
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)

    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    tau = np.linalg.norm(g_xyz_equat_body)*ut.INV_C

    # Second try using tau
    #M = ob.calc_M(jd-tau, body.tp_jd, body.a)
    M = calc_M(jd-tau, body.tp_jd, body.a)
    M = reduce_rad(M,True)
    xyz, _ = elliptic_orbit(next_E, M, body.a, body.e)
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)    
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    return polarFcartesian(g_xyz_equat_body)
"""

def calc_orbits_data(eph, planets, minor_bodys, comets):
    orbs = {}
    dfs = []
    for name in planets:
        print ("Calculating data for ",name)
        df = calc_eph_planet(name, eph)
        orbs[name] = df
        dfs.append(df)
    for obj in minor_bodys:
        if isinstance(obj, BodyElems) : 
            body_elms = obj
        else :
            body_elms = read_body_elms_for(obj, DF_BODYS)
        print ("Calculating data for ",body_elms.name)
        df  = calc_eph_minor_body_perturbed(body_elms,eph)
        orbs[body_elms.name] = df
        dfs.append(df)        
    for name in comets:
        comet_elms = read_comet_elms_for(name, DF_COMETS)
        print ("Calculating data for ",comet_elms.name)
        df  = calc_eph_minor_body_perturbed(comet_elms,eph)
        orbs[name] = df
        dfs.append(df)        

    date_refs = orbs[first(orbs.keys())]['date'].to_list()
    cols=['h_x','h_y','h_z']    
    for k, df in orbs.items():
        orbs[k] = df[cols].to_numpy() 
    
    return orbs, df, date_refs


def test_calc_eph_comet():
    HALLEY_1950 = dc.CometElms(name="1P/Halley",
                epoch_name=None ,
                q =  0.5870992 ,
                e = 0.9672725 ,
                i_dg = 162.23932 ,
                Node_dg = 58.14397 ,
                w_dg = 111.84658 ,
                tp_str = "19860209.44",
                equinox_name = "B1950")

    """
    ENCKE = CometElms(name="2P/Encke",
            epoch_name=None ,
            q =  2.2091404*(1-0.8502196) ,
            e = 0.8502196 ,
            i_dg = 11.94524 ,
            Node_dg = 334.75006 ,
            w_dg = 186.23352 ,
            tp_str = "19901028.54502",
            equinox_name = "J2000")                

    
    elm1 = dc.CometElms(name="Halley",
                q = 0.5870992,
                e = 0.9673,
                i_dg = 162.2384,
                Node_dg = 58.154,
                w_dg = 111.857,
                tp_str = "19860209.44",                
                equinox_name = "1986.02.09.44")
    """



    C_1988 = dc.read_comet_elms_for('C/1988 L1 (Shoemaker-Holt-Rodriquez)', dc.DF_COMETS)

    eph_c1988 = EphemrisInput(from_date="1980.01.01.0",
                        to_date = "2020.01.01.0",
                        step_dd_hh_hhh = "100 00.0",
                        equinox_name = "J2000")    

    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)
    
    
    eph1 = EphemrisInput(from_date="2020.10.15.0",
                        to_date = "2020.12.25.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")
    #print (dc.APOFIS)

    #df = calc_eph_minor_body_perturbed(dc.APOFIS, eph)

    eph_halley = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")

    eph_encke = EphemrisInput(from_date="1990.10.6.0",
                        to_date = "1991.01.28.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")


    eph3 = EphemrisInput(from_date="2020.10.15.0",
                        to_date = "2020.12.25.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")                        
    
    eph4 = EphemrisInput(from_date="1984.1.1",
                        to_date = "1984.2.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")           



    df = calc_eph_comet(dc.HALLEY_B1950, eph_halley)
    print (df.head(20)[df.columns[0:8]])

    #df = calc_eph_comet(ENCKE, eph_encke)
    #print (df.head(20))




def test_perturbed():


    #D_1993 = dc.read_comet_elms_for('D/1993 F2-A', dc.DF_COMETS)

    # Parabolic
    C_1988 = dc.read_comet_elms_for('C/1988 L1 (Shoemaker-Holt-Rodriquez)', dc.DF_COMETS)

    eph_c1988 = EphemrisInput(from_date="1988.01.01.0",
                        to_date = "1988.02.01.0",
                        step_dd_hh_hhh = "01 00.0",
                        equinox_name = "J2000")    

    # Hyperbolic
    C_1980 = dc.read_comet_elms_for('C/1980 E1 (Bowell)', dc.DF_COMETS)
    eph_c1980 = EphemrisInput(from_date="1980.01.01.0",
                        to_date = "1981.02.01.0",
                        step_dd_hh_hhh = "01 00.0",
                        equinox_name = "J2000")    



    # Read Orbital elements for Ceres
    
    CERES = dc.BodyElems(name="Ceres",
                   epoch_name="1983.09.23.0",
                   a = 2.7657991,
                   e = 0.0785650,
                   i_dg = 10.60646,
                   Node_dg = 80.05225,
                   w_dg = 73.07274,
                   M_dg = 174.19016,
                   equinox_name = "1950.0")
    
    CERES = dc.read_body_elms_for("Ceres",dc.DF_BODIES)
    #print (CERES)

    """
    eph = EphemrisInput(from_date="2017.06.27.0",
                        to_date = "2017.07.25.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")
    """

    eph = EphemrisInput(from_date="1994.01.01.0",
                    to_date = "1994.04.01.0",
                    step_dd_hh_hhh = "3 00.0",
                    equinox_name = "J2000")

    eph_halley = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")

    
    #print ("Ephemeris Data:")
    
    
    eph_ceres = EphemrisInput(from_date="1992.06.27.0",
                        to_date = "1992.10.25.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")

    eph_ceres = EphemrisInput(from_date="2020.06.16.0",
                        to_date = "2039.07.31.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")

    
    #print (dc.APOFIS)


    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)


    #df = calc_eph_minor_body_perturbed(C_1988, eph_c1988)
    df = calc_eph_minor_body_perturbed(C_1988, eph_c1988)
    #print (df.head(20))
    print (df[df.columns[0:8]])


def test_planets():

    eph = EphemrisInput(from_date="2017.06.27.0",
                        to_date = "2017.07.25.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")


    print (calc_eph_planet("Pluto",eph))

if __name__ == "__main__":
    test_calc_eph_comet()
    #test_perturbed()
    ##test_planets()
