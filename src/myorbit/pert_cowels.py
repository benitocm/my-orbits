""" This module containst the function to calculate the ephemeris of minor bodys following the
perturbation method by Cowels
"""
# Standard library imports
import logging

# Third party imports
import numpy as np
from numpy.linalg import norm
# Using Newton-Ramson method
from scipy.integrate import solve_ivp    

# Local application imports
from myorbit import coord as co
import myorbit.coord as co
from myorbit.util.general import frange
import myorbit.orbutil as ob
import myorbit.data_catalog as dc
from myorbit.ephemeris_input import EphemrisInput
from myorbit.kepler.keplerian import KeplerianStateSolver
from myorbit.util.constants import *

logger = logging.getLogger(__name__)


def my_dfdt(t, y):        
    """Computes the time derivative of the unknown function. Integrating this function, we obtain the unknown
    function. We know the velocity and acceleration that is basically what this function returns so integrating we obtain 
    the position and velocity.
    This method basically calculates the acceleration based on the position (r). This acceleration is 
    composed of the one corresponding to the force exerted by the Sun (two-body acceleration) plus
    the perturbed one due to the every planet that which includes two components:
        - The direct one: The force the planet exerts to the body
        - The indirect one: The force the planet exerts to the Sun that also impacts to the body.

    Parameters
    ----------
    t : float
        point in time (normally used in modified julian days) at which we want to calculate the derivative
    y : np.array
        The state vector (np.arrayy[6]) where [0..3] is the position vector r and [3..6] is the velocity 
        vector (in this case is not used)

    Returns
    -------
    np.array
        A vector of 6 positions (np.array[6]) where [0..3] is the velocity vector v and [3..6] is the
        aceleration vector
    """

    h_xyz = y[0:3]
    acc = -ob.accel(GM, h_xyz) + ob.calc_perturbed_accelaration(t, h_xyz)
    return np.concatenate((y[3:6], acc))


def calc_eph_by_cowells (body, eph , obj_type='body', include_osc=False):
    """Computes the ephemeris for a minor body using the Cowells method. This has 
    more inexact than Enckes but quicker

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

    # The PQR mtx (from orbital plane to eclipt) is preccesed from equinox of body object to the desired equinox 
    MTX_J2000_PQR = co.mtx_eclip_prec(body.T_eqx0, T_J2000).dot(body.mtx_PQR)
    
    # The initial conditions for doing the integration is calculated, i.e.,
    # the r,v of the body at its epoch (in the example of Ceres, the epoch of 
    # book that epoch  is 1983/09/23.00)
    # The integration is done in the ecliptic plane and precessed in to J2000
    # so the solution will be also ecliptic and precessed.

    initial_mjd = body.epoch_mjd  
    if hasattr(body, 'q') :
        # Comets
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, q= body.q, a=body.a, epoch=None, M_at_epoch=None)
    else :
        # Asteroids 
        solver = KeplerianStateSolver.make(tp_mjd = body.tp_mjd, e=body.e, a=body.a, epoch=body.epoch_mjd, M_at_epoch=body.M0)     
     
    xyz0, vxyz0, *other =  solver.calc_rv(initial_mjd)
    
    y0 = np.concatenate((MTX_J2000_PQR.dot(xyz0), MTX_J2000_PQR.dot(vxyz0)))  
    
    # The integration is done in the ecliptic plane. First, we propagete the state vector
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
        sol_1 = ob.do_integration(my_dfdt, y0, initial_mjd, t_sol[-1], t_sol)
        # Second one forwards
        t_sol = list(frange(initial_mjd, eph.to_mjd, eph.step))
        sol_2 = ob.do_integration(my_dfdt, y0, initial_mjd, t_sol[-1], t_sol)

        SOL_T = np.concatenate((sol_1.t, sol_2.t))
        SOL_Y = np.concatenate((sol_1.y, sol_2.y), axis=1)
    else :
        t_sol = list(frange(eph.from_mjd, eph.to_mjd, eph.step))
        if eph.to_mjd < initial_mjd :
            # If the epoch is in the future, we need to integrate backwards, i.e.
            # propagating the state vector from the future to the past.
            t_sol = list(reversed(t_sol))
        sol = ob.do_integration(my_dfdt, y0, initial_mjd, t_sol[-1], t_sol)       
        SOL_T = sol.t
        SOL_Y = sol.y

    tpoints = dict()
    for idx, t in enumerate(SOL_T) :  
        tpoints[t] = (SOL_Y[:,idx][:3], SOL_Y[:,idx][3:6])
    tpoints = {t:tpoints[t] for t in sorted(tpoints.keys())}

    return ob.process_solution(tpoints, MTX_J2000_Teqx, MTX_equatFeclip, eph.eqx_name)


def test_all():
    eph = EphemrisInput(from_date="2020.05.25.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    for name in dc.DF_BODIES['Name']: 
        body = dc.read_body_elms_for(name,dc.DF_BODIES)
        print ("Calculating for ",name)
        print (calc_eph_by_cowells(body, eph)) 



def test_body():
    eph = EphemrisInput(from_date="2020.05.25.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")

    eph = ob.EphemrisInput(from_date="2019.04.01.0",
                       to_date = "2020.07.01.0",
                       step_dd_hh_hhh = "10 00.0",
                       equinox_name = "J2000")

    """
    eph = ob.EphemrisInput(from_date="2020.04.01.0",
                       to_date = "2020.07.01.0",
                       step_dd_hh_hhh = "50 00.0",
                       equinox_name = "J2000")
    """
    
    B_2002_NN4 = dc.read_body_elms_for("2002 NN4",dc.DF_BODYS)


    CERES = dc.read_body_elms_for("Ceres",dc.DF_BODYS)


    df = calc_eph_by_cowells(B_2002_NN4, eph)

    print (df[df.columns[0:8]])

    

def test_comet():

    eph  = EphemrisInput(from_date="1985.11.15.0",
                        to_date = "1986.04.05.0",
                        step_dd_hh_hhh = "10 00.0",
                        equinox_name = "J2000")

    HALLEY_J2000 = dc.read_comet_elms_for("1P/Halley", dc.DF_COMETS)


    df = calc_eph_by_cowells(HALLEY_J2000, eph, 'comet')

    print (df[df.columns[0:8]])



def test_all_comets():
    """
    eph = EphemrisInput(from_date="2020.05.25.0",
                        to_date = "2020.06.15.0",
                        step_dd_hh_hhh = "02 00.0",
                        equinox_name = "J2000")
    """


    df_ = dc.DF_COMETS
    #df_ = df_.query("0.999 < e < 1.001")
    df_ = df_.query("Name=='C/1532 R1'")

    for idx, name in enumerate(df_['Name']): 
        logger.warning(f"{idx+1} Calculating for {name} ")
        body = dc.read_comet_elms_for(name,dc.DF_COMETS)
        eph = EphemrisInput.from_mjds( body.epoch_mjd-25, body.epoch_mjd+25, "02 00.0", "J2000" )
        print (f"{idx+1} Calculating for {name} ")
        print (calc_eph_by_cowells(body, eph, 'comet')) 

def test_several_comets():
    names = [#"C/1988 L1 (Shoemaker-Holt-Rodriquez)",   # Parabolic
             #"C/1980 E1 (Bowell)"]  
             #"C/1848 P1 (Petersen)"
              "C/1672 E1"]

    for name in names :
        body = dc.read_comet_elms_for(name,dc.DF_COMETS)
        eph = EphemrisInput.from_mjds( body.epoch_mjd-50, body.epoch_mjd+50, "02 00.0", "J2000" )
        print (f"Calculating for {name} ")
        df = calc_eph_by_cowells(body, eph, 'comet')
        print (df[df.columns[0:8]])
        #print (calc_eph_by_enke(body, eph, 'comet')) 

    

if __name__ == "__main__" :
    #test_all()
    #test_2()
    #test_all_comets()
    test_comet()
    #test_several_comets()
    # test_body()
    #test_comet()


 



    





    

