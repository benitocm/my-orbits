"""
This module contains functions related to orbit calculations
"""
# Standard library imports
import logging
from time import  process_time
from math import isclose

# Third party imports
import numpy as np
from numpy.linalg import norm
import toolz as tz
from scipy.integrate import solve_ivp    

# Local application imports
from . import orbutil as ob
from . import coord as co
from .lagrange.lagrange_coeff import calc_rv_from_r0v0
from .kepler.keplerian import KeplerianStateSolver
from .util.general import mu_Sun, my_range, measure, pow
from .init_config import ENKES_ABS_TOL

logger = logging.getLogger(__name__)

from numba import jit

def f1(vector):
    # Utility function
    return vector/pow(norm(vector),3)

def calc_F(c, b, abs_tol=ENKES_ABS_TOL):
    """According to the book Orbital Mechanics for Engineering, this is the 
    recommended way to solve the equation F=1-c^3/b^3 when c/b is aproximately 1.
    That equation shows up in the Enkes approach.
    This documented in the Appendix F of the book.
    I have tested with several tolerances and I have not seen any improvements 

    Parameters
    ----------
    c : float
        
    b : float
        
    Returns
    -------
    float
        The solution for 1-c^3/b^3
    """
    if isclose(c/b, 1.0, rel_tol=0., abs_tol=abs_tol) :
        a = b-c
        q = (2*b*a-pow(a,2))/pow(b,2)
        return (pow(q,2)-3*q+3)*q/(1+((1-q)*np.sqrt(1-q)))
    else :
        return 1 - pow(c/b,3)

def my_dfdt(t, y, r0_xyz, v0_xyz, t0):
    """Computes the time derivative of the unknown function. Integrating this function, we obtain the unknown
    function. We know the velocity and acceleration that is basically what this function returns so integrating we obtain 
    the position and velocity.

    Parameters
    ----------
    t : float
        Time of computation (normally used in Modified Julian days) 
    y : np.array[6]
        The vector with the variables to solve the differential equation system
                [0..3] delta_r
                [3..6] delta_v (not used in this case)
    r0_xyz : np.array[3] 
            Initial radio vector (cartesian) from the Sun to the body with respect to the
            orbital plane (perifocal frame) [AU]        
    v0_xyz : np.array[3]
            Initial velocity vector (cartesian) of the body with respect to the orbital 
            plane (perifocal frame) [AU/days]
    t0 : [type]
        Initial time (normally in Modified Julian Days)

    Returns
    -------
    np.array[6]
        A vector vector of 6 positions where:
            [0..2] is the  delta_v 
            [3..6] is the  delta_acc 
    """
    deltar_xyz = y[0:3]    
    # The two-bodys orbit is calculated starting at r0, v0 and t-t0 as elapsed time
    rosc_xyz, *other = calc_rv_from_r0v0(mu_Sun, r0_xyz, v0_xyz, t-t0)    
    # The radio vector perturbed is the two-bodys plus the delta_r
    rpert_xyz = rosc_xyz + deltar_xyz
    F = calc_F(norm(rosc_xyz), norm(rpert_xyz))
    delta_acc = (-mu_Sun/pow(norm(rosc_xyz),3))*(deltar_xyz- F*rpert_xyz)+ob.calc_perturbed_accelaration(t, rpert_xyz)    
    return np.concatenate((y[3:6],delta_acc))
  
@measure
def apply_enckes(eph, t_range, r0_xyz, v0_xyz):
    """Utility function needed because the integration needs to be done in two intervals so this function
    is called for each of these intervals. It applies the enckles's approach, i.e. calculates the dr and dv
    to modified the two bodys (osculating orbit)

    Parameters
    ----------
    eph : EphemrisInput
        Ephemeris data 
    t_range : np.array[]
             A numpy vector with the time samples where each time sample defines a time interval.
             The enckles method is applied in each one of this interval. The time samples are 
             modified julian days.
    r0_xyz : np.array[3]
            Initial radio vector (cartesian) from the Sun to the body with respect to the
            orbital plane (perifocal frame) [AU]        
    v0_xyz : np.array[3]
            Initial velocity vector (cartesian) of the body with respect to the orbital 
            plane (perifocal frame) [AU/days]

    Returns
    -------
    dict
        A dictionary where the key is a time reference in days (modified julian days) and the 
        the value is the a tuple with two vectors, the radio vector r and the velocity vector at the time reference
    """
    steps = np.diff(t_range)
    result = dict()
    clock_mjd = t_range[0]
    ms_acc = 0
    for idx, step in enumerate(steps) :
        t1 = int(round(process_time() * 1000))
        sol = solve_ivp(my_dfdt, (clock_mjd, clock_mjd+step), np.zeros(6), args=(r0_xyz, v0_xyz, clock_mjd) , rtol = 1e-12)  
        ms_acc += (int(round(process_time() * 1000)) - t1)
        assert sol.success, "Integration was not OK!"
        rosc_xyz, vosc_xyz, *other = calc_rv_from_r0v0 (mu_Sun, r0_xyz, v0_xyz, step)
        # The last integration value is taken
        r0_xyz = rosc_xyz + sol.y[:,-1][:3]
        v0_xyz = vosc_xyz + sol.y[:,-1][3:6]
        # If the clock is in the middle of the ephemeris time, it is inserted in the solution
        if eph.from_mjd <= clock_mjd+step <= eph.to_mjd :
            result[clock_mjd+step] = (r0_xyz, v0_xyz)
        clock_mjd += step    
    logger.debug (f"Total Elapsed time for solve_ivp: {ms_acc} ms ")
    return result 

def calc_eph_by_enckes (body, eph, include_osc=False):
    """Computes the ephemeris for a minor body using the Enckes method. This has more precission that
    the Cowells but it takes more time to be calculated.

    Parameters
    ----------
    body : CometElms, BodyElms
        Orbital elements of the body which ephemeris is desired to calculate. In case of the
        body is a comet, the type of this parameter must be CometElms. In case of the boyd is a small body
        the type of this parameter must be BodyElms.
    eph :EphemrisInput
        The ephemeris data

    Returns
    -------
    pd.DataFrame
        Dataframe with the ephemeris data calculated.
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

    # In the ecliptic.
    r0 = MTX_J2000_PQR.dot(xyz0)
    v0 = MTX_J2000_PQR.dot(vxyz0)

    if eph.from_mjd < initial_mjd < eph.to_mjd :
        # If the epoch is in the middle, we need to integrate forward and backwards
        #t_range = list(ut.frange(initial_mjd, eph.to_mjd, eph.step))
        t_range = my_range(initial_mjd, eph.to_mjd, eph.step)
        result_1 = apply_enckes(eph, t_range, r0, v0)

        # and backwards 
        #t_range = list(ut.frange(eph.from_mjd, initial_mjd, eph.step))
        #if t_range[-1] != initial_mjd :
        #    t_range.append(initial_mjd)
        t_range = list(reversed(my_range(eph.from_mjd, initial_mjd, eph.step)))
        result_2 = apply_enckes(eph, t_range, r0, v0)
        solution = tz.merge([result_1,result_2])        
        
    elif initial_mjd < eph.from_mjd :
        """
        # If the epoch is in the past, we need to integrate forward
        t_range_1 = list(ut.frange(initial_mjd, eph.from_mjd, eph.step))
        # The previous ensure that initial_mjd is included but the eph.from  may be not included
        # so we test the final value to know if we need to include manually
        if t_range_1[-1] != eph.from_mjd :
            t_range_1.append(eph.from_mjd)
        """
        # [initial, from] 
        t_range_1 = my_range(initial_mjd, eph.from_mjd-eph.step, eph.step)
        # [from+step, to]
        t_range_2 = my_range(eph.from_mjd, eph.to_mjd, eph.step, include_end=False)

        """
        t_range_2 = list(ut.frange(eph.from_mjd+eph.step, eph.to_mjd, eph.step))
        if len(t_range_2) == 0 :
            t_range_2.append(eph.to_mjd)
        if t_range_2[-1] != eph.to_mjd :
            t_range_2.append(eph.to_mjd)
        """
        solution = apply_enckes(eph, t_range_1 + t_range_2, r0, v0)
        #Only the t's in the from-to is included
        solution = tz.keyfilter(lambda k : k in t_range_2, solution)
    else :
        # If the epoch is in the future, we need to integrate backwards
        # goes from the epoch backward toward the end value from 
        # the ephemeris and inital value of the ephemeris

        #[initial_mjd ---> backwards to  --> eph.to.mjd]
        t_range_1 = my_range(eph.to_mjd, initial_mjd, eph.step)

        """
        t_range_1 = list(ut.frange(eph.to_mjd, initial_mjd, eph.step))
        # The previous ensure that eph.to is included but the initial may be not included
        # so we test the final value to know if we need to include manually
        if t_range_1[-1] != initial_mjd :
            t_range_1.append(initial_mjd)
        """
        """
        t_range_2 = list(ut.frange(eph.from_mjd, eph.to_mjd, eph.step))
        # the previous ensure that eph.from is included but the to_mjd may be included
        # but  we include in the previous so we need to remove it . We test the last element to check
        # if we need to remove it
        if t_range_2[-1] == eph.to_mjd :
            t_range_2 = t_range_2[0:-1]
        t_range = list(reversed(t_range_1)) + list(reversed(t_range_2))
        """
        t_range_2 = my_range(eph.from_mjd, eph.to_mjd, eph.step, include_end=False)        
        t_range = list(reversed(t_range_2+t_range_1))
        solution = apply_enckes(eph, t_range, r0, v0)
        #Only the t's in the from-to is included
        solution = tz.keyfilter(lambda k : k in t_range_2, solution)

    solution = {t:solution[t] for t in sorted(solution.keys())}
        
    return ob.process_solution(solution, MTX_J2000_Teqx, MTX_equatFeclip, eph.eqx_name, include_osc)


if __name__ == "__main__" :
    None


 



    





    

