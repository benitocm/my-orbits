""" This module containst the function to calculate the ephemeris of minor bodys following the
perturbation method by Cowels
"""
# Standard library imports
import logging

# Third party imports
import numpy as np
from numpy.linalg import norm

# Local application imports
from . import coord as co
from . import orbutil as ob
from .util.general import frange
from .kepler.keplerian import KeplerianStateSolver
from .util.general import mu_Sun

logger = logging.getLogger(__name__)

def my_dfdt(t, y, mu=mu_Sun):               
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
    acc = -ob.accel(mu, h_xyz) + ob.calc_perturbed_accelaration(t, h_xyz)
    return np.concatenate((y[3:6], acc))


def calc_eph_by_cowells (body, eph, include_osc=False):
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
    return ob.process_solution(tpoints, MTX_J2000_Teqx, MTX_equatFeclip, eph.eqx_name, include_osc)


if __name__ == "__main__" :
    None


 



    





    

