""" This module contains utility functions to keplerian orbit calculations
"""
# Standard library imports
import logging

# Third party imports
import numpy as np
from numpy import  arctan2, sqrt, sqrt, rad2deg, deg2rad
import pandas as pd
import toolz as tz
from numpy.linalg import norm

# Using Newton-Ramson method
from scipy.integrate import solve_ivp    

# Local application imports
from . import coord as co
from . import data_catalog as dc
from .util import timeut as tc
from .coord import mtx_eclip_prec
from .util.timeut  import  MDJ_J2000, JD_J2000, CENTURY, mjd2jd
from .util.general import pow, mu_Sun
from .planets import h_xyz_eclip_eqxdate, h_xyz_eclip_pluto_j2000, g_rlb_eclip_sun_eqxdate, g_xyz_equat_sun_j2000
from .util.constants import INV_C, PI, GM_by_planet, GM

logger = logging.getLogger(__name__)

def accel(gm, r_xyz):
    """Computes the acceleration based on the corresponding GM of the planet and the
    radio vector of the body 

    Parameters
    ----------
    gm : float
        Gravitational constant times the mass of the planet 
    r_xyz : np.array[3]
        Radio vector of the body

    Returns
    -------
    np.array[3]
        The acceleration vector
    """
    return gm*r_xyz/pow(norm(r_xyz),3) 


def calc_perturbed_accelaration(t_mjd, he_xyz_eclip_body) :
    """Computes the acceleration vector for a minor body in the solar system at one point
    time. The perturbation caused by the planets (including Pluto) to the minor body 
    are taking into account

    Parameters
    ----------
    t_mjd : float
        Time of computation [Modified Julian Day]
    h_xyz_eclip_body : np.array[3]
        Heliocentric cartesian coordinates of the minor body

    Returns
    -------
    float
        The acceleration vector
    """

    # The century T corresponding to the time t. Also used to calculate
    # precession matrix
    T = (t_mjd - MDJ_J2000)/CENTURY    
    # desired equinox is J2000, so T_desired is 0
    T_desired = (JD_J2000 - JD_J2000)/CENTURY
    mtx_prec = mtx_eclip_prec(T, T_desired)
    acc = 0
    for pl_name, pl_GM in GM_by_planet.items() :
        if pl_name == 'Pluto':
            he_xyz_eclipt_planet = h_xyz_eclip_pluto_j2000(mjd2jd(t_mjd))
        else :
            # Planetary position (ecliptic and equinox of J2000)
            he_xyz_eclipt_planet = mtx_prec.dot(h_xyz_eclip_eqxdate(pl_name, mjd2jd(t_mjd)))

        he_xyz_planet_body = he_xyz_eclip_body - he_xyz_eclipt_planet
        # Direct accelaration
        acc += accel(pl_GM, he_xyz_planet_body)
        # Indirect acceletration
        acc += accel(pl_GM, he_xyz_eclipt_planet)
    return -acc    


def my_f(t, Y, mu=mu_Sun):       
    """Derivated function that will be numerically integrated. It will contain the 
    velocity vector and the acceleration vector. Once this is integrated numerically,
    the state vector is obtained, i.e, the position vector and the velocity vector.

    Parameters
    ----------
    t : float
        Time of computation
    Y : np.array[6]
        [0..2] the radio vector 
        [3..5] the velocity vector
    mu : float, optional
        The gravitational constant times the mass of the Sun, by default mu_Sun

    Returns
    -------
    np.array[6]
        [0..2] the velocity vector 
        [3..5] the acceleration vector including the one due to the Sun and the ones due to 
            the planets.
    """
    he_xyz = Y[0:3]
    # - Sun acceleration  - perturbed acceleration
    acc = -accel(mu, he_xyz) + calc_perturbed_accelaration(t, he_xyz)
    return np.concatenate((Y[3:6], acc))


def do_integration(fun_t_y, y0 , t_begin, t_end, t_samples):
    """Integrates numerically from t_begin to t_end the function vector with the velocities and the acelerations
    to obtain the state vector, (position and velocity) of the body. The integration is solved
    as a Initial Value problem, i.e, a Y0 vector is provided with the initial state vector (r0_xyz, v0_xyz)

    Parameters
    ----------
    fun_t_y : function
        The derivative function that we want to integrate to obtain the posisiton of the body 
    y0 : np.array[6]
        Initial state vector of the body:
            [0..2] the initial radio vector of the body
            [3..5] the initial velocity of the body
    t_begin : float
        Lower limit of the definite integral 
    t_end : float
        Higher limit of the definite integral 
    t_samples : int
        Number of samples to obtain

    Returns
    -------
    sol
        The solution of the IVP problem provided by the solve_ivp method 
    """
    sol = solve_ivp(fun_t_y, (t_begin,t_end), y0, t_eval=t_samples, rtol = 1e-12)  
    if sol.success :
        return sol
    else :
        logger.warn("The integration was failed: "+sol.message)


def calc_osculating_orb_elmts(h_xyz, h_vxyz, epoch_mjd=0, equinox="J2000",mu=mu_Sun):
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
    a = 1.0/(2.0/R-v_2/mu)
    e_cosE = 1.0-R/a
    e_sinE = h_xyz.dot(h_vxyz)/sqrt(mu*np.abs(a))
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
    """Utitly method used in the calculation of the ephemeris to obtain the equatorial geocentric coordinates
    of the objects. The result of the ephemeris calculation is time series with position and veolocity
    vectors of the body referred to the ecliptic plane, i.e., the PQR matrix has been applied to the
    orbital plane. So this method takes that data and do the common things to convert them into 
    geocentric coordinates to provide the final result.    

    Parameters
    ----------
    tpoints : dict
            A dictionary where the key is the time (in modified julian dates) and the value is
            a tuple with the position vector and velocity vector of the body referred to the
            ecliptic, i.e., eclipic heliocentric coordinates of the object. The position is in
            AU and the velocity and AUs/days
    MTX_J2000_Teqx : np.array[3][3]
            Matrix (3x3) to change from the J2000 equinox to the one requested by the Ephemeris input
    MTX_equatFeclip : np.array[3][3]
             Matrix (3x3) to change from ecliptic to equaatorial system always to be applied
                to cartesian or rectangular coordinates
    eph_eqx_name : str
        [description]
    include_osc : bool, optional
        Flat to indicate whether to include the osculating elements in the solution, by default False

    Returns
    -------
    pd.DataFrame
        A dataframe witht the solution 
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

        if include_osc :
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
    if include_osc :
        cols += oscul_keys     
    return df[cols].sort_values(by='t_mjd')

if __name__ == "__main__":
    None