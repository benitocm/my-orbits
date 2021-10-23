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
import sys

# Third party imports
import pandas as pd
import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan2, arctan, sqrt,cosh,sinh,deg2rad,rad2deg, sqrt, arcsinh
from numpy.linalg import norm
import toolz as tz
# Using Newton-Ramson method
from scipy.optimize import newton, bisect
from scipy.integrate import solve_ivp    
from toolz import pipe, compose
from myorbit.cluegen import Datum
import logging


# Local application imports
#from myastro import util as ut
#from myastro import timeutil as tc
#from myastro import coord as co
#from myastro import planets as pl
#from myastro import data_catalog as dc


#from myastro.timeutil import  PI_HALF, PI, TWOPI, MDJ_J2000, JD_J2000, CENTURY, mjd2jd, reduce_rad
#from myastro.coord import Coord, as_str, EQUAT2_TYPE, ECLIP_TYPE
#from myastro.util import pow, GM
#from myastro.data_catalog import DF_PLANETS, BodyElems
#from myastro.planets import h_rlb_eclip_eqxdate, h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000, g_rlb_eclip_sun_eqxdate, g_xyz_equat_sun_j2000

#from myastro.timeutil import epochformat2jd, jd2mjd, T, mjd2jd, jd2str_date

from myorbit.util.timeut import MDJ_J2000, CENTURY, JD_J2000, reduce_rad
import myorbit.util.timeut as tc
import myorbit.coord as co
import myorbit.data_catalog as dc
from myorbit.planets import h_rlb_eclip_eqxdate, h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000, g_rlb_eclip_sun_eqxdate, g_xyz_equat_sun_j2000
from myorbit.util.constants import *

logger = logging.getLogger(__name__)


def calc_osculating_orb_elmts(h_xyz, h_vxyz, epoch_mjd=0, equinox="J2000"):
    """ 
    Computes the orbital elements of an elliptical orbit from position
    and velocity vectors
    
    Args:
        r : Heliocentric ecliptic position vector  in [AU]
        v : Heliocentric ecliptic velocity vector  in [AU/d]
        epoch_mjd : time in Modified Julian Day where the orbital elements are calculalted 
                    i.e., the epoch
        
    Returns :
        An OrbElmtsData 
    """
    h = np.cross(h_xyz, h_vxyz)
    H = norm(h)
    Omega = arctan2(h[0], -h[1])
    i = arctan2(sqrt(pow(h[0],2)+pow(h[1],2)),h[2])
    u = arctan2(h_xyz[2]*H, -h_xyz[0]*h[1]+h_xyz[1]*h[0])
    R = norm(h_xyz)
    v_2 = h_vxyz.dot(h_vxyz)
    a = 1.0/(2.0/R-v_2/GM)
    e_cosE = 1.0-R/a
    e_sinE = h_xyz.dot(h_vxyz)/sqrt(GM*a)
    e_2 = pow(e_cosE,2) + pow(e_sinE,2)
    e = sqrt(e_2)
    E = arctan2(e_sinE,e_cosE)
    M = E - e_sinE
    nu = arctan2(sqrt(1.0-e_2)*e_sinE, e_cosE-e_2)
    omega = u - nu
    if Omega < 0.0 :
        Omega += 2.0*PI
    if omega < 0.0 :
        omega += 2.0*PI
    if M < 0.0:
        M += 2.0*PI    
    return dc.BodyElms.in_radians("Osculating Body",tc.mjd2epochformat(epoch_mjd),a,e,i,Omega,omega,M,equinox)


def process_solution(tpoints, MTX_J2000_Teqx, MTX_equatFeclip):
    """ 
    Utitly method used in the calculation of the ephemeris to obtain the equatorial geocentric coordinates
    of the objects. The result of the ephemeris calculation is time series with position and veolocity
    vectors of the body referred to the ecliptic plane, i.e., the PQR matrix has been applied to the
    orbital plane. So this method takes that data and do the common things to convert them into 
    geocentric coordinates to provide the final result.    
    
    Args:
        tpoints : A dictionary where the key is the time (in modified julian dates) and the value is
                  a tuple with the position vector and velocity vector of the body referred to the
                  ecliptic, i.e., eclipic heliocentric coordinates of the object. The position is in
                  AU and the velocity and AUs/days
        MTX_J2000_Teqx : Matrix (3x3) to change from the J2000 equinox to the one requested by the Ephemeris input
        MTX_equatFeclip : Matrix (3x3) to change from ecliptic to equaatorial system always to be applied
                          to cartesian or rectangular coordinates
        
    Returns :
        A DataFrame with an entry per each tpoint with the required cols
    """

    cols = ['date','Sun(dg)','h_l','h_b','h_r','ra','dec','r[AU]','h_x','h_y','h_z','t_mjd']
    rows = []
    for t, r_v in tpoints.items():
        clock_mjd = t
        row = {}
        clock_jd = tc.mjd2jd(clock_mjd)
        row['date'] = tc.jd2str_date(clock_jd)
        row['t_mjd'] = clock_mjd

        # Because the integration is done in the ecliptic coordinate system and precessed
        # so the solution will be already ecliptic so we just need to precess it to the 
        # desired equinox before transforming it to the equatorial

        h_xyz = MTX_J2000_Teqx.dot(r_v[0])
        h_vxyz = MTX_J2000_Teqx.dot(r_v[1])

        h_xyz_equat_body = MTX_equatFeclip.dot(h_xyz)
        h_vxyz_equat_body = MTX_equatFeclip.dot(h_vxyz)

        # This is just to obtain the geo ecliptic longitud of the Sun and include in the
        # dataframe. Becasue the we have the sun position at equinox of the date,
        # we need to preccess it (in ecplitpical form) to J2000
        T = (clock_jd - tc.JD_J2000)/tc.CENTURY
        g_rlb_eclipt_T = tz.pipe (g_rlb_eclip_sun_eqxdate(clock_jd, tofk5=True) , 
                               co.cartesianFpolar, 
                               co.mtx_eclip_prec(T,tc.T_J2000).dot, 
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
        g_xyz_equat_body -= INV_C*norm(g_xyz_equat_body)*h_vxyz_equat_body

        g_rlb_equat_body = co.polarFcartesian(g_xyz_equat_body)        

        row['ra'] = co.format_time(tz.pipe(g_rlb_equat_body[1], rad2deg,  tc.dg2h, tc.h2hms))
        row['dec'] = co.format_dg(*tz.pipe(g_rlb_equat_body[2], rad2deg, tc.dg2dgms))
        row['r[AU]'] = r_AU
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[cols].sort_values(by='t_mjd')

class ElemsData (Datum):
    i: float        #Inclination (rads)
    Omega : float   #Ascending node (rads)
    omega : float   #Argument of perihelion (rads) 
    p : float       #Semilatus rectum (AU)
    e : float       #Eccentricity (AU)
    q : float       #Perihelion distance (AU)
    a : float       #Semi-major axis (AU) 
    n : float       #mean monition 
    nu : float       #True anomaly (rads)
    tp_mjd : float  #Time of perihelion passage (modified Julian day)

def show_orbital_elements(elem):
    print (f"Inclination: {rad2deg(elem.i)} dg.")
    print (f"Longitude of ascending node: {rad2deg(elem.Omega)} dg.")
    print (f"Longitude of perihelion: {rad2deg(elem.Omega+elem.omega)} dg.")
    print (f"Argument of perihelion: {rad2deg(elem.omega)} dg.")
    print (f"Semilatus rectum : {elem.p} AU.")
    print (f"Eccentricity : {elem.e} AU.")
    print (f"Perihelion distance : {elem.q} AU.")
    print (f"Semi-major axis : {elem.a} AU.")
    print (f"Mean motion : {elem.n} AU.")
    print (f"True anomaly: {rad2deg(elem.nu)} dg.")
    print (f"Time of perihelion passage: {tc.mjd2epochformat(elem.tp_mjd)}           {tc.mjd2jd(elem.tp_mjd)} Julian day")


def f (eta, m, l, max_iter = 30):
    """
    Helper function to be used in find_eta function
    """
    w = m/pow(eta,2)-l
    if abs(w) < 0.1 :
        # Series expansion        
        W = 4.0/3.0
        term = W
        for n in range (1, max_iter):
            term *= w*(n+2.0)/(n+1.5)
            W += term
            if isclose(abs(term), 0, abs_tol=1e-07) :
                break
        logger.error(f"Not converged after {max_iter} iterations")
    elif w > 0.0 :
        g = 2.0 * arcsin(sqrt(w))
        W = (2*g -sin(2.0*g))/pow(sin(g),3)
    else :
        g = 2.0 * arcsinh(sqrt(-w))
        W = (sinh(2.0*g)-2.0*g)/pow(sinh(g),3)
    return  1 - eta + (w+l)*W
    
    
def find_eta (r_a, r_b, tau, max_iter=30, epsilon=1e-7) :
    """
    Computes the sector-triangle ratio for two given positions and the time bettween them
    Args:
        r_a = Position vector at first intant in [AU]
        r_b = Position vector at second instant in [AU]
        tau = time between the positions r_a and r_b (kGaus * dt in [days])
    Returns:
        A float withe sector-triangle ratio        
    """    
    s_a = norm(r_a)
    s_b = norm(r_b)
    kappa = sqrt(2.0*(s_a*s_b+r_a.dot(r_b)))
    m = pow(tau,2)/pow(kappa,3)
    l = (s_a+s_b)/(2.0*kappa) - 0.5
    eta_min = sqrt(m/(1.0+l))
    # Start with Hansen's approximation
    eta2 = ( 12.0 + 10.0*sqrt(1.0+(44.0/9.0)*m /(l+5.0/6.0)) ) / 22.0
    eta1 = eta2 + 0.1
    # Secant method
    f1 = f(eta1, m, l)
    f2 = f(eta2, m, l)
    i = 0
    while not isclose(f1, f2, abs_tol=epsilon) :
        d_eta =  -f2*(eta2-eta1)/(f2-f1)
        eta1 = eta2
        f1 = f2
        while (eta2+d_eta <= eta_min):
            d_eta *= 0.5
        eta2 += d_eta
        f2 = f(eta2, m, l)
        i += 1
        if i == max_iter:
            logger.error(f"Not converged after {max_iter} iterations and epsilon {epsilon}")
            break
    return eta2 


def calc_orb_elms_from_2rs(mu, h_xyz_a, ta_mjd, h_xyz_b, tb_mjd) :
    """
    Computes the orbital elements for body given two positions of it.
    Args:
        GM : Product of gravitation constant and centre of mass [AU^3/d^2]
        r_a : Heliocentric eclipctical position A in [AU]
        ta_mjd : Time of passage of position A (Modified Julian day)
        r_b : Heliocentric eclipctical position B in [AU]
        tb_mjd : Time of passage of position A (Modified Julian day)
    Returns:
        An ElemData class with the orbital Elements
    """    
    elem = ElemsData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    norm_a = norm(h_xyz_a)
    e_a = h_xyz_a/norm_a
    norm_b = norm(h_xyz_b)
    fac = h_xyz_b.dot(e_a)
    # Calculate r0 (fraction of h_xyz_b perpendicular to h_xyz_a)
    h_xyz_0 = h_xyz_b - fac*e_a
    norm_0 = norm(h_xyz_0)
    e_0 = h_xyz_0/norm_0
    # Inclination and ascending node
    R = np.cross(e_a, e_0)
    R_polar = co.polarFcartesian(R)  # r, phi, theta
    elem.i = PI_HALF - R_polar[2]
    elem.Omega = reduce_rad(PI_HALF+R_polar[1], to_positive=True)       
    if elem.i == 0.0 :
        u = arctan2(h_xyz_a[1],h_xyz_a[0])
    else :
        u = arctan2(e_0[0]*R[1]-e_0[1]*R[0], -e_a[0]*R[1]+e_a[1]*R[0])
    # Semilatus rectum
    tau = sqrt(mu) * abs(tb_mjd - ta_mjd)
    eta = find_eta(h_xyz_a, h_xyz_b, tau)
    elem.p = pow(norm_a * norm_0 *eta/tau, 2)
    # Eccentricity, true anomaly and argument of perihelion
    cos_dnu = fac/norm_b
    sin_dnu = norm_0/norm_b
    ecos_nu = elem.p/norm_a - 1.0
    esin_nu = (ecos_nu * cos_dnu - (elem.p/norm_b-1.0))/sin_dnu
    elem.e = sqrt(pow(ecos_nu,2)+pow(esin_nu,2))
    elem.nu = reduce_rad(arctan2(esin_nu, ecos_nu), to_positive=True)    
    elem.omega = reduce_rad(u-elem.nu, to_positive=True)    
    # Perihelion distance, semi-major axis and mean motion
    elem.q = elem.p/(1.0+elem.e) 
    elem.a = elem.q/(1.0-elem.e)
    elem.n = sqrt(mu/abs(pow(elem.a,3)))
    # Mean anomaly and time of perihelion passage
    if elem.e < 1.0 :
        E = arctan2(sqrt((1.0-elem.e)*(1.0+elem.e)) * esin_nu, ecos_nu + pow(elem.e,2))
        elem.M = E - elem.e*sin(E)
    else :
        sinhH = sqrt((elem.e-1.0)*(elem.e+1.0)) *  esin_nu/(elem.e+elem.e*ecos_nu)
        elem.M = elem.e * sinhH - np.log (sinhH + sqrt(1.0+pow(sinhH,2)))
    elem.M = reduce_rad(elem.M , to_positive=True)   
    elem.tp_mjd = ta_mjd - elem.M / elem.n
    return elem

def calc_orb_elms_from_rv(mu, r, v) :
    """
    Computes the orbital elements for body given the position and the velocity
    Args:
        mu  : 
        r  : Heliocentric eclipctical position A in [AU]
        v  : 
    Returns:
        An ElemData class
    """    
    elem = ElemsData(0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0)
    h = np.cross(r,v)
    K = np.array([0,0,1])
    n_v = np.cross(K,h)
    e_v = ((pow(norm(v),2) - mu/norm(r))*r - r.dot(v)*v)/mu
    elem.e = norm(e_v)
    xi = v.dot(v)/2 - (mu/norm(r))
    if isclose(elem.e, 1.0, abs_tol=1e-4) :
        elem.a = 0
        elem.p = pow(norm(h),2)/mu
    else :
        elem.a = -mu/(2*xi)
        elem.p = elem.a* (1-pow(elem.e,2))
    # Inclination 
    elem.i = arccos(h[2]/norm(h))
    elem.Omega = arccos(n_v[0]/norm(n_v))
    if n_v[1] < 0 :
        elem.Omega = TWOPI - elem.Omega
    elem.omega = arccos(n_v.dot(e_v)/(norm(n_v)*norm(e_v)))
    if e_v[2] < 0 :
        elem.omega = TWOPI - elem.omega
    elem.nu = arccos(e_v.dot(r)/(norm(e_v)*norm(r)))
    if r.dot(v) < 0 :
        elem.nu = TWOPI - elem.nu
    return elem



if __name__ == "__main__" :

    r1 = np.array([ 0.68279753,  2.59192608, -0.05022929])
    t1 = -19430.009884937575
    r2 = np.array([-1.85787079,  1.69259011,  0.3937317 ])
    t2 = -19170.16713014274
    v2 = r2-r1/(t2-t1)
    show_orbital_elements(calc_orb_elms_from_rv(GM,r2,v2))





    

  



    





    

