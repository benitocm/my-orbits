""" This module contains utility functions to keplerian orbit calculations
"""
# Standard library imports
from functools import partial
from math import isclose
import logging

# Third party imports
import numpy as np
from numpy import  arctan2, sqrt, sqrt
import pandas as pd
import toolz as tz
from numpy.linalg import norm

# Using Newton-Ramson method
from scipy.optimize import newton, bisect
from scipy.integrate import solve_ivp    

# Local application imports
from myorbit import coord as co
import myorbit.util.timeut as tc
import myorbit.data_catalog as dc
from myorbit.util.timeut  import  MDJ_J2000, JD_J2000, CENTURY, mjd2jd
from myorbit.util.general import pow
from myorbit.planets import h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000, g_rlb_eclip_sun_eqxdate, g_xyz_equat_sun_j2000
from myorbit.util.constants import *


logger = logging.getLogger(__name__)

def do_iterations(func, x_0, abs_tol = 1e-08, max_iter=50):
    """General function to iterate mathematical functions of one variable

    Parameters
    ----------
    func : function
        The mathematical funtion of one variable to iterate
    x_0 : float
        Initial value of the variable
    abs_tol : float, optional
        [description], by default 1e-08
    max_iter : int, optional
        Maximum number of iterations, by default 50

    Returns
    -------
    tuple (boolean, float, int)
        The result of the iteration where:
            boolean param, indicates whether the solution converged or not
            float param, is the final value of the variable
            n, is the last iteration done
    """
    x = x_0
    for n in range(max_iter):
        new_value = func(x)
        if isclose(new_value, x, abs_tol = abs_tol) :
            return (True,new_value,n)
        else :
            x = new_value
    return (False,new_value,n)

def solve_ke(e, func_e_anomaly,  m_anomaly):
    """Solves the kepler equation, i.e., calculates the excentric anomaly

    Parameters
    ----------
    e : float
        The eccentricity [0,1]
    func_e_anomaly : function
        Function to calculate the excentric anomaly used to iterate
    m_anomaly : float
        Mean anomaly in angle units (rads)

    Returns
    -------
    float
        The value fo the excentric anomaly (if converged). Otherwise, return None        
    """

    f = partial(func_e_anomaly,e,m_anomaly)
    res = do_iterations(f,m_anomaly,100)
    if not res[0] :
        logger.error("Not converged")
        return 
    else :
        logger.debug(f"Converged in {res[2]} iterations wih result {np.rad2deg(res[1])} degrees")
        return res[1]


def solve_ke_newton(e, func_e_anomaly, m_anomaly, e_anomaly_0=None):
    """ Solves the kepler equation, i.e., calculates the excentric anomaly by using
     the Newton-Ranson method

    Parameters
    ----------
    e : float
        eccentricity [0,1]
    func_e_anomaly : function
        function to calculate the excentric anomaly used to iterate
    m_anomaly : float
        Mean anomaly in angle units (radians)
    e_anomaly_0 : float, optional
        Initial value to start the iteration, by default None

    Returns
    -------
    float
        The value fo the excentric anomaly (if converged). Otherwise, return None        
    """
  
    f = partial(func_e_anomaly , e ,m_anomaly)
    x, root = newton(lambda x : f(x) - x, e_anomaly_0, fprime=None, tol=1e-12, maxiter=50, fprime2=None, full_output=True)    
    #logger.error(root)
    if not root.converged :
        logger.error(f"Not converged: {root}")
    return x

def solve_ke_bisect(e,m_anomaly,func_e_anomaly):
    """[summary]

    Parameters
    ----------
    e : [type]
        [description]
    m_anomaly : [type]
        [description]
    func_e_anomaly : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    m_rad = np.deg2rad(m_anomaly)    
    f = partial(func_e_anomaly,e,m_rad)    
    x = bisect(f, 0, 0.25, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, full_output=False, disp=True)
    return x


def accel(gm, r):
    """Computes the acceleration based on the corresponding GM and the
    radio vector (distance to the Sun)

    Parameters
    ----------
    gm : float
       The product G*M (Gravity constant * Sun Mass)
    r : float
        Distance to the Sun [AU]

    Returns
    -------
    float
        The acceleration 
    """
    return gm*r/pow(norm(r),3) 


def calc_perturbed_accelaration(t_mjd, h_xyz_eclip_body ) :
    """Computes the acceleration vector for a minor body in the solar system at one point
    time. The perturbation caused by the planets (including Pluto) to the minor body 
    are taking into account

    Parameters
    ----------
    t_mjd : float
        Point in time as Modified Julian Day 
    h_xyz_eclip_body : np.array[3]
        Heliocentric cartesian coordinates of the minor body

    Returns
    -------
    float
        The acceleration
    """

    # The century T corresponding to the time t. Also used to calculate
    # precession matrix
    T = (t_mjd - MDJ_J2000)/CENTURY    
    # desired equinox is J2000, so T_desired is 0
    T_desired = (JD_J2000 - JD_J2000)/CENTURY
    mtx_prec = co.mtx_eclip_prec(T, T_desired)
    acc = 0
    #for pl_name in filter(lambda x : (x != 'Sun') and (x != 'Pluto'), GM_by_planet.keys()) :
    for pl_name in filter(lambda x : (x != 'Sun') , GM_by_planet.keys()) :
        if pl_name == 'Pluto':
            h_xyz_eclipt_planet = h_xyz_eclip_pluto_j2000(mjd2jd(t_mjd))
        else :
            # Planetary position (ecliptic and equinox of J2000)
            h_xyz_eclipt_planet = mtx_prec.dot(h_xyz_eclip_eqxdate(pl_name, mjd2jd(t_mjd)))

        h_xyz_planet_body = h_xyz_eclip_body - h_xyz_eclipt_planet
        # Direct accelaration
        acc += accel(GM_by_planet[pl_name], h_xyz_planet_body)
        # Indirect acceletration
        acc += accel(GM_by_planet[pl_name], h_xyz_eclipt_planet)
    return -acc    

def my_f(t, Y):        
    h_xyz = Y[0:3]

    # - Sun acceleration  - perturbed acceleration
    acc = -accel(GM, h_xyz) + calc_perturbed_accelaration(t, h_xyz)

    #def calc_accelaration(t_mjd, h_xyz_eclip_body ) :
    #acc = calc_accelaration(t,h_xyz)
    return np.concatenate((Y[3:6], acc))


def do_integration(fun_t_y, y0 , t_begin, t_end, t_samples):
    sol = solve_ivp(fun_t_y, (t_begin,t_end), y0, t_eval=t_samples, rtol = 1e-12)  
    if sol.success :
        return sol
    else :
        logger.warn("The integration was failed: "+sol.message)


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
    Omega = np.arctan2(h[0], -h[1])
    i = arctan2(sqrt(pow(h[0],2)+pow(h[1],2)),h[2])
    u = arctan2(h_xyz[2]*H, -h_xyz[0]*h[1]+h_xyz[1]*h[0])
    R = norm(h_xyz)
    v_2 = h_vxyz.dot(h_vxyz)
    a = 1.0/(2.0/R-v_2/GM)
    e_cosE = 1.0-R/a
    e_sinE = h_xyz.dot(h_vxyz)/sqrt(GM*np.abs(a))
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
       
def process_solution(tpoints, MTX_J2000_Teqx, MTX_equatFeclip, eph_eqx_name, include_osc=False):
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
    oscul_keys = []
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

        oscul_elms = calc_osculating_orb_elmts(h_xyz, h_vxyz, clock_mjd, eph_eqx_name)
        row.update(oscul_elms.as_dict())
        if len(oscul_keys) == 0 :
           oscul_keys = list(oscul_elms.as_dict().keys())


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
        row['Sun(dg)'] = f"{tc.rad2deg(g_rlb_eclipt_T[1]):03.1f}"
        
        g_xyz_equat_sun = g_xyz_equat_sun_j2000(clock_jd)

        #We need to log the heliocentirc ecliptic coordinates in polar format 
        rlb = co.polarFcartesian(h_xyz)
        row['h_l'] = f"{tc.rad2deg(rlb[1]):03.1f}"
        row['h_b'] = f"{tc.rad2deg(rlb[2]):03.1f}"
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

        row['ra'] = co.format_time(tz.pipe(g_rlb_equat_body[1], tc.rad2deg,  tc.dg2h, tc.h2hms))
        row['dec'] = co.format_dg(*tz.pipe(g_rlb_equat_body[2], tc.rad2deg, tc.dg2dgms))
        row['r[AU]'] = r_AU
        rows.append(row)

    df = pd.DataFrame(rows)
    if include_osc :
        cols += oscul_keys     
    return df[cols].sort_values(by='t_mjd')
