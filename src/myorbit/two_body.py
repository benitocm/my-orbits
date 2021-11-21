""" This module provides the functionality to calculate ephemeris for two bodies problem
also in the case of perturbed methods. More advance perturbed methods will be handled  
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
import myorbit.orbutil as ob
import myorbit.data_catalog as dc
from myorbit.util.general import frange, mu_Sun
from myorbit.util.constants import INV_C
from myorbit.init_config import H_ABS_TOL, EC_ABS_TOL
from myorbit.util.timeut import CENTURY, JD_J2000, dg2h, h2hms, dg2dgms, T_given_mjd, mjd2jd, jd2str_date
from myorbit.planets import g_xyz_equat_sun_j2000
from myorbit.kepler.keplerian import KeplerianStateSolver
from myorbit.kepler.ellipitical import calc_rv_for_elliptic_orbit, calc_M
from myorbit.lagrange.lagrange_coeff import calc_rv_from_r0v0
from myorbit.data_catalog import read_planet_elms_for


logger = logging.getLogger(__name__)



def calc_eph_planet_by_keplerian(body, eph): 
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
    
    # Normally, the equinox of the data of the body will be J2000  and the equinox of the 
    # ephemeris will also be J2000, so the precession matrix will be the identity matrix
    # Just in the case of book of Personal Astronomy with your computer pag 81 is used   
    MTX_Teqx_PQR = co.mtx_eclip_prec(body.T_eqx0, eph.T_eqx).dot(body.mtx_PQR)  

    # Transform from ecliptic to equatorial just depend on desired equinox
    MTX_equatFecli = co.mtx_equatFeclip(eph.T_eqx)

    if hasattr(body, 'q') :
        # Comets
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, q= body.q, a=body.a, epoch=None, M_at_epoch=None, force_orbit=force_orbit)
    else :
        # Asteroids 
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, a=body.a, epoch=body.epoch_mjd, M_at_epoch=body.M0, force_orbit=force_orbit)     

    result = dict()
    # Angular momentums in the orbit
    hs = []
    # Eccentricy vector
    es = []
    for clock_mjd in frange(eph.from_mjd, eph.to_mjd, eph.step):        
        r_xyz, v_xyz, r, h_xyz, e_xyz, *other = solver.calc_rv(clock_mjd)
        hs.append(h_xyz)
        es.append(e_xyz)
        result[clock_mjd] = (MTX_Teqx_PQR.dot(r_xyz), MTX_Teqx_PQR.dot(v_xyz))
    if not all(np.allclose(h_xyz, hs[0], atol=H_ABS_TOL) for h_xyz in hs):
        msg = f'The angular momentum is NOT constant in the orbit'
        logger.error(msg)
    if not all(np.allclose(e_xyz, es[0], atol=EC_ABS_TOL) for e_xyz in es):
        msg = f'The eccentricy vector is NOT constant in the orbit'
        logger.error(msg)

    return ob.process_solution(result, np.identity(3), MTX_equatFecli, eph.eqx_name, False)
    
    



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
    

def calc_eph_twobody(body, eph, force_orbit=None):
    """ Computes the ephemeris for a small body or comet

    Parameters
    ----------
    body : CometElms, BodyElms
        Orbital elements of the body which ephemeris is desired to calculate. In case of the
        body is a comet, the type of this parameter must be CometElms. In case of the boyd is a small body
        the type of this parameter must be BodyElms.
    eph : EphemrisInput
        The entry data of the ephemeris

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

    if hasattr(body, 'q') :
        # Comets
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, q= body.q, a=body.a, epoch=None, M_at_epoch=None, force_orbit=force_orbit)
    else :
        # Asteroids 
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, a=body.a, epoch=body.epoch_mjd, M_at_epoch=body.M0, force_orbit=force_orbit)     

    result = dict()
    # Angular momentums in the orbit
    hs = []
    # Eccentricy vector
    es = []
    for clock_mjd in frange(eph.from_mjd, eph.to_mjd, eph.step):        
        r_xyz, v_xyz, r, h_xyz, e_xyz, *other = solver.calc_rv(clock_mjd)
        hs.append(h_xyz)
        es.append(e_xyz)
        result[clock_mjd] = (MTX_Teqx_PQR.dot(r_xyz), MTX_Teqx_PQR.dot(v_xyz))
    if not all(np.allclose(h_xyz, hs[0], atol=H_ABS_TOL) for h_xyz in hs):
        msg = f'The angular momentum is NOT constant in the orbit'
        logger.error(msg)
    if not all(np.allclose(e_xyz, es[0], atol=EC_ABS_TOL) for e_xyz in es):
        msg = f'The eccentricy vector is NOT constant in the orbit'
        logger.error(msg)

    return ob.process_solution(result, np.identity(3), MTX_equatFecli, eph.eqx_name, False)

def calc_eccentricity_vector(r_xyz, rdot_xyz, h_xyz, mu=mu_Sun):
    """[summary]

    Parameters
    ----------
    r_xyz : [type]
        [description]
    rdot_xyz : [type]
        [description]
    h_xyz : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    return  (np.cross(rdot_xyz,h_xyz) - (mu*r_xyz/np.linalg.norm(r_xyz)))/mu

def calc_eph_twobody_universal(body, eph):
    """ Computes the ephemeris for a small body or comet using the Universal approach

    Parameters
    ----------
    body : CometElms, BodyElms
        Orbital elements of the body which ephemeris is desired to calculate. In case of the
        body is a comet, the type of this parameter must be CometElms. In case of the boyd is a small body
        the type of this parameter must be BodyElms.
    eph : EphemrisInput
        The entry data of the ephemeris

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

    if hasattr(body, 'q') :
        # Comets
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, q= body.q, a=body.a, epoch=None, M_at_epoch=None)
    else :
        # Asteroids 
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, a=body.a, epoch=body.epoch_mjd, M_at_epoch=body.M0)     

    # A Keplerian solver is needed to calculate the r0_xyz and the r0dot_xyz, i.e., the initial radio vector and
    # the velocity vector at time t0, in this case t0 is eph.from_mjd . Once we have them, the orbit propagation 
    # is done using the Universal variable approach
    result = dict()
    r0_xyz, r0dot_xyz, r0, h0_xyz, *others = solver.calc_rv(eph.from_mjd)
    result[eph.from_mjd] = (MTX_Teqx_PQR.dot(r0_xyz), MTX_Teqx_PQR.dot(r0dot_xyz))
    # List Angular momentums in the orbit
    hs = []
    # List of eccentricy vector in the orbit
    es = []
    for clock_mjd in frange(eph.from_mjd+eph.step, eph.to_mjd, eph.step):  
        r_xyz, rdot_xyz, h_xyz, f = calc_rv_from_r0v0(mu_Sun, r0_xyz, r0dot_xyz, clock_mjd-eph.from_mjd)   
        e_xyz = calc_eccentricity_vector(r_xyz, rdot_xyz, h_xyz)
        hs.append(h_xyz)
        es.append(e_xyz)
        result[clock_mjd] = (MTX_Teqx_PQR.dot(r_xyz), MTX_Teqx_PQR.dot(rdot_xyz))
    if not all(np.allclose(h_xyz, hs[0], atol=1e-12) for h_xyz in hs):
        msg = f'The angular momentum is NOT constant in the orbit'
        print (msg)
        logger.error(msg)
    if not all(np.allclose(e_xyz, es[0], atol=1e-12) for e_xyz in es):
        msg = f'The eccentricy vector is NOT constant in the orbit'
        print (msg)
        logger.error(msg)

    return ob.process_solution(result, np.identity(3), MTX_equatFecli, eph.eqx_name, False)




def calc_eph_minor_body_perturbed (body, eph ,include_osc=False):
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
    if hasattr(body, 'q') :
        # Comets
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, q= body.q, a=body.a, epoch=None, M_at_epoch=None)
    else :
        # Asteroids 
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, a=body.a, epoch=body.epoch_mjd, M_at_epoch=body.M0)     
        
    xyz0, vxyz0, *other =  solver.calc_rv(initial_mjd)
     
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


#
# Currently this code is not being used but I want to keep it because the same idea is used 
# in other parts, the sum of vectors are done in the equatorial system rather than int the ecliptic system
# 

def _g_rlb_equat_body_j2000(jd, body):    
    """Computes the geocentric polar coordinates of body (eliptical orbit) based 
    at particular time and orbital elements of the body following the second 
    algorithm of the Meeus (this is the one that explains that the sum of the vectors 
    are done in the equatorial system and it works better than doing in the eliptic system)

    Parameters
    ----------
    jd : float
        Time of computation in Julian days        
    body : BodyElms
        Elemnents of the body whose position is calculated

    Returns
    -------
    np.array:
        A 3-vector with the geocentric coordinates of the body in Equatorial system
    """
   
    T_J2000 = 0.0 # desired equinox

    # The matrix will include the precesion from equinox of the date of the body
    mtx_equat_PQR =  np.linalg.multi_dot([co.mtx_equatFeclip(T_J2000),
                               co.mtx_eclip_prec(body.T_eqx0,T_J2000),
                               co.mtx_gauss_vectors(body.Node, body.i, body.w)                                                          
                               ])

	# The equatorial xyz of the sun referred to equinox J2000 for the 
	# specific moment
    g_xyz_equat_sun = g_xyz_equat_sun_j2000(jd)    
    
    # Fist calculation of the position of the body is done
    #M = calc_M(jd, body.tp_jd, body.a)

    M = calc_M (jd, body.tp_jd, body.a)
    xyz, *others = calc_rv_for_elliptic_orbit(M, body.a, body.e)

	# xyz are cartesians in the orbital plane (perifocal system) so we need to transform 
	# to equatorial
    		
    # This is key to work. The sum of the vectors is done in the equatorial system.
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    tau = np.linalg.norm(g_xyz_equat_body)*INV_C

    # Second try using tau
    M = calc_M (jd-tau, body.tp_jd, body.a)
    xyz, *others = calc_rv_for_elliptic_orbit(M, body.a, body.e)


    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)    
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    return co.polarFcartesian(g_xyz_equat_body)

if __name__ == "__main__":
    from myorbit.ephemeris_input import EphemrisInput
    body = read_planet_elms_for("Jupiter",dc.DF_PLANETS)
    eph = EphemrisInput(from_date="2000.01.01.0",
                    to_date = "2000.04.01.0",
                    step_dd_hh_hhh = "10 00.0",
                    equinox_name = "J2000")
    cols=["date","t_mjd","h_x","h_y","h_z","ra","dec"]
    df = calc_eph_twobody(body, eph)
    df = df[cols]
    print (df)
    print ("with vsop")
    df = calc_eph_planet("Jupyter",eph)
    df = df[cols]
    print(df)
    
    
