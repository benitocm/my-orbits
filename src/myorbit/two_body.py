""" This module provides the functionality to calculate ephemeris for two bodies problem
also in the case of perturbed methods. More advance pertubed methods will be handled  
in other module
"""

# Standard library imports
import logging

# Third party imports
import pandas as pd
import numpy as np
from numpy.linalg import norm
from toolz import pipe

# Local application imports
import myorbit.planets as pl
from myorbit import coord as co
from myorbit.util.general import frange
from myorbit.util.timeut import CENTURY, JD_J2000, dg2h, h2hms, dg2dgms, T_given_mjd, mjd2jd, jd2str_date
from myorbit.orbits.keplerian import KeplerianOrbit
import myorbit.orbits.orbutil as ob
import myorbit.data_catalog as dc
from myorbit.orbits.ephemeris_input import EphemrisInput

from myorbit.util.constants import *

logger = logging.getLogger(__name__)


def calc_eph_planet(name, eph): 
    """Computes the ephemeris for a planet based on the VSOP87 theory.

    Parameters
    ----------
    name : str
        Name of the planet
    eph : EphemrisInput
        The entry data of the ephemeris

    Returns
    -------
    pd.DataFrame
        Dataframe with the ephemeris data calculated.
    """
    rows = []
    for clock_mjd in frange(eph.from_mjd, eph.to_mjd, eph.step):        
        row = {}
        clock_jd = mjd2jd(clock_mjd)
        row['date'] = jd2str_date(clock_jd)
        row['t_mjd'] = clock_mjd

        #T = (clock_mjd - MJD_J2000)/CENTURY    
        T = T_given_mjd(clock_mjd)
        # desired equinox is J2000, so T_desired is 0
        T_desired = (JD_J2000 - JD_J2000)/CENTURY
        mtx_prec = co.mtx_eclip_prec(T, T_desired)

        if name.lower() == 'pluto':
            h_xyz_eclipt = pl.h_xyz_eclip_pluto_j2000(clock_jd)            
            g_rlb_equat = pl.g_rlb_equat_pluto_j2000(clock_jd)
        else :
            # Planetary position (ecliptic and equinox of J2000)
            h_xyz_eclipt = mtx_prec.dot(pl.h_xyz_eclip_eqxdate(name, clock_jd))
            g_rlb_equat = pl.g_rlb_equat_planet_J2000(name,clock_jd)
        
        row['h_x'] = h_xyz_eclipt[0]
        row['h_y'] = h_xyz_eclipt[1]
        row['h_z'] = h_xyz_eclipt[2]

        row['ra'] = co.format_time(pipe(g_rlb_equat[1], np.rad2deg,  dg2h, h2hms))
        row['dec'] = co.format_dg(*pipe(g_rlb_equat[2], np.rad2deg, dg2dgms))
        row['r [AU]'] = norm(h_xyz_eclipt)
 
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(by='t_mjd')
    

def calc_eph_twobody(body, eph, obj_type='comet'):
    """ Computes the ephemeris for a small body or comet

    Parameters
    ----------
    body : CometElms, BodyElms
        Orbital elements of the body which ephemeris is desired to calculate. In case of the
        body is a comet, the type of this parameter must be CometElms. In case of the boyd is a small body
        the type of this parameter must be BodyElms.
    eph : EphemrisInput
        The entry data of the ephemeris
    obj_type : str, optional
        Type of the object ('body' for small bodies or 'comet' for comets), by default 'comet'

    Returns
    -------
    pd.DataFrame
        Dataframe with the ephemeris data calculated.
    """

    # Normally, the equinox of the data of the body will be J2000  and the equinox of the 
    # ephemeris will also be J2000, so the precession matrix will be the identity matrix
    # Just in the case of book of Personal Astronomy with your computer pag 81 is used   
    MTX_Teqx_PQR = co.mtx_eclip_prec(body.T_eqx0, eph.T_eqx).dot(body.mtx_PQR)  

    # Transform from ecliptic to equatorial just depend on desired equinox
    MTX_equatFecli = co.mtx_equatFeclip(eph.T_eqx)

    if obj_type == 'comet' :
        k_orbit = KeplerianOrbit.for_comet(body)
    else :
        k_orbit = KeplerianOrbit.for_body(body)
     
    result = dict()
    for clock_mjd in frange(eph.from_mjd, eph.to_mjd, eph.step):        
        r , v = k_orbit.calc_rv(clock_mjd)
        result[clock_mjd] = (MTX_Teqx_PQR.dot(r), MTX_Teqx_PQR.dot(v))
    return ob.process_solution(result, np.identity(3), MTX_equatFecli, eph.eqx_name, False)


def calc_eph_minor_body_perturbed (body, eph , type, include_osc=False):
    """ Computes the ephemeris for a small body or comet taking into account
    the perturbations introduced by the planets.

    Parameters
    ----------
    body : CometElms, BodyElms
        Orbital elements of the body which ephemeris is desired to calculate. In case of the
        body is a comet, the type of this parameter must be CometElms. In case of the boyd is a small body
        the type of this parameter must be BodyElms.
    eph : EphemrisInput
        The entry data of the ephemeris
    obj_type : str, optional
        Type of the object ('body' for small bodies or 'comet' for comets), by default 'comet'        
    include_osc: boolean, optional
        Flag to indicate whether the osculating elements should be included or not in the final result

    Returns
    -------
    pd.DataFrame
        Dataframe with the ephemeris data calculated. It will include the osculating elements
        according to the include_osc parameter.
    """
        
    # This matrix just depends on the desired equinox to calculate the obliquity
    # to pass from ecliptic coordinate system to equatorial
    MTX_equatFeclip = co.mtx_equatFeclip(eph.T_eqx)

    T_J2000 = 0.0    
    # This is to precess from J2000 to ephemeris equinox (normally this matrix will be identity matrix)
    MTX_J2000_Teqx = co.mtx_eclip_prec(T_J2000,eph.T_eqx)

    # The PQR mtx (from orbital plane to eclipt) is precessed from equinox of body object to the desired equinox 
    MTX_J2000_PQR = co.mtx_eclip_prec(body.T_eqx0, T_J2000).dot(body.mtx_PQR)
    
    # The initial conditions for doing the integration is calculated, i.e.,
    # the r,v of the body at its epoch (in the example of Ceres, the epoch of 
    # book that epoch  is 1983/09/23.00)
    # The integration is done in the ecliptic plane and precessed in to J2000
    # so the solution will be also ecliptic and precessed.

    if body.epoch_mjd is None :
        print ("This body has not an epoch data so we don't know when the the orbital data were calculated")
        return pd.DataFrame()

    initial_mjd = body.epoch_mjd  

    if type == 'comet' :
        k_orbit = KeplerianOrbit.for_comet(body)
    else :
        k_orbit = KeplerianOrbit.for_body(body)

    xyz0, vxyz0 =  k_orbit.calc_rv(initial_mjd)
     
    y0 = np.concatenate((MTX_J2000_PQR.dot(xyz0), MTX_J2000_PQR.dot(vxyz0)))  
    
    # The integration is done in the ecliptic plane. First, we propagate the state vector
    # from the epoch of the body (when the data are fresh) up to the start of the ephemeris
    # in that interval we don't request the solution. 
    # Second we integrate in the ephemeris interval and asking the solution every step 
    # (e.g. 2 days) this is t.sol time samples
    # In case the epoch of the objet is at the future, the integration is done backward
    # in time (during the same interval but in reverse mode)

    if eph.from_mjd < initial_mjd < eph.to_mjd :
        # We need to do 2 integrations
        # First one backwards
        t_sol = list(reversed(list(frange(eph.from_mjd, initial_mjd, eph.step))))
        sol_1 = ob.do_integration(ob.my_f, y0, initial_mjd, t_sol[-1], t_sol)
        # Second one forwards
        t_sol = list(frange(initial_mjd, eph.to_mjd, eph.step))
        sol_2 = ob.do_integration(ob.my_f, y0, initial_mjd, t_sol[-1], t_sol)

        SOL_T = np.concatenate((sol_1.t, sol_2.t))
        SOL_Y = np.concatenate((sol_1.y, sol_2.y), axis=1)
    else :
        t_sol = list(frange(eph.from_mjd, eph.to_mjd, eph.step))
        if eph.to_mjd < initial_mjd :
            # If the epoch is in the future, we need to integrate backwards, i.e.
            # propagatin the state vector from the future to the past.
            t_sol = list(reversed(t_sol))
        sol = ob.do_integration(ob.my_f, y0, initial_mjd, t_sol[-1], t_sol)       
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

    result = dict()
    for idx, clock_mjd in enumerate(SOL_T) :  
        result[clock_mjd] = (SOL_Y[:,idx][:3],SOL_Y[:,idx][3:6])
    return ob.process_solution(result, MTX_J2000_Teqx, MTX_equatFeclip, eph.eqx_name, True)

    

def test_jupiter():
    # TODO Add test for Planets including Pluto

    eph = EphemrisInput(from_date="2017.06.27.0",
                        to_date = "2017.07.25.0",
                        step_dd_hh_hhh = "2 00.0",
                        equinox_name = "J2000")

    print (calc_eph_planet("Jupiter",eph))



if __name__ == "__main__":
    test_jupiter()
    