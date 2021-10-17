"""This module contains functions related to the calculation of Keplerian orbits.

"""
# Standard library imports
from math import isclose
import logging

# Third party imports
import numpy as np
from numpy import sqrt, cos, sin, cosh, sinh, tan, arctan
from numpy.linalg import multi_dot

# Local application imports
from myorbit.util.timeut import reduce_rad
from myorbit.orbits.orbutil import solve_ke_newton
from myorbit.planets import g_xyz_equat_sun_j2000
from myorbit.util.general import pow
from myorbit import coord as co

from myorbit.util.constants import *

logger = logging.getLogger(__name__)

def calc_M_for_body(t_mjd, epoch_mjd, period_in_days, M_at_epoch) :
    """ 
    Computes the mean anomaly based on the data of BodyElms, in this case,
    uses the period (calculated) and the Mean anomaly at epoch.
    For Body Elements

    Args:
        t_mjd : time of the computation as Modified Julian Day
        t0: a point in time where Mean anomaly is knwon
        M0: the mean anomaly in radians at t0
        period : period of the orbit in days.

    Returns :
        The mean anomaly in radians
    """
    M = (t_mjd - epoch_mjd)*TWOPI/period_in_days
    M += M_at_epoch
    return reduce_rad(M,to_positive=True)     


def calc_M_for_comet(t_mjd, tp_mjd, inv_a) -> float :
    """ 
    Computes the mean anomaly as a function a t, t0 and a, i.e., not depending on the
    period of the orbit=1) and semi-major axis

    Args:
        t : time of the computation in Modified Julian Dates
        t0: Time of perihelion passage 
        a: semi-major axis in AU

    Returns :
        The mean anomaly in radians
    """
    #M = sqrt(ut.GM)*(t_mjd-tp_mjd)/np.float_power(a, 1.5)
    M = sqrt(GM)*(t_mjd-tp_mjd)*sqrt(pow(inv_a, 3))
    return reduce_rad(M, to_positive=True)


def calc_M_by_E(e, E):
    """ 
    Computes the true anomaly 

    Args:
        e : eccentricity of the orbit
        e_anomaly : eccentric anomaly in radians
        
    Returns :
        The true anomaly in angle units (radians) 
    """

    return 2 * arctan(sqrt((1+e)/(1-e))* tan(E/2))    


def next_E (e:float, m_anomaly:float, E:float) -> float :
    """
    Iterative function to calculate the eccentric anomaly, i.e.,
    computes the next eccentric value for ellictical orbits.
    Used in the Newton method (Pag 65 of Astronomy of 
    the personal computer)

    Args:
        e : eccentricity 
        m_anomaly: Mean anomaly in angle units (rads)
        E : Previous value of the eccentric anomaly

    Returns :
        The eccentric anomaly in angle units (radians)
    """

    num = E - e * sin(E) - m_anomaly
    den = 1 - e * cos(E)
    return E - num/den


def elliptic_orbit (next_E_func, m_anomaly, a, e):
    """ 
    Computes position (r) and velocity (v) vectors for elliptic orbits
    Pg 66 of Astronomy on the Personal computer
    The m_anomaly is the one that varies with time so the result
    of this function will also vary with time.

    An ellipse has (0<e<1)

    Args:
        next_E_func : iterating function to calculate the eccentric anomaly, used
                      for solving kepler equation with Newton 
        m_anomaly : The current Mean anomaly in rads (it will depend on time t)
        a : Semi-major axis of the orbit in [AU]
        e  : Eccentricity of the orbit (<1 for elliptical)
        
    Returns :
        Position (r) w.r.t. orbital plane in [AU] 
        Velocity (v) w.r.t orbital plane in [AU/days]

    """

    E0 = m_anomaly if (e<=0.8) else PI

    e_anomaly = solve_ke_newton(e, next_E_func, m_anomaly, E0)
    # Pg 65 Astronomy on the Personal Computer
    #t_anomaly = calc_M_by_E(e,e_anomaly)
    
    cte = sqrt(GM/a)
    cos_E = cos(e_anomaly)
    sin_E = sin(e_anomaly)
    fac =  sqrt(1-e*e)
    rho = 1.0 - e*cos_E
    r =  np.array([a*(cos_E-e), a*fac*sin_E, 0.0]) #x,y at pag 62
    v =  np.array([-cte*sin_E/rho, cte*fac*cos_E/rho, 0.0])
    return r,v

def next_H (e:float, mh_anomaly: float, H:float) -> float:
    """
    Iterative function to calculate the eccentric anomaly, i.e.,
    computes the next eccentric value for hyperbolic orbits.
    Used in the Newton method (Pag 65 of Astronomy of 
    the personal computer)

    Args:
        e : eccentricity 
        mh_anomaly: Mean anomaly in angle units (rads) 
        H : Previous value of the eccentric anomaly

    Returns :
        The eccentric anomaly in angle units (radians)
    """
    num = e * sinh(H) - H - mh_anomaly
    den = e * cosh(H) - 1
    return H - num/den

def hyperpolic_orbit (tp, next_H_func, a, e, t):
    """ 
    Computes position (r) and velocity (v) vectors for hiperbolic orbits
    Pg 66 of Astronomy on the Personal computer
    The m_anomaly is the one that varies with time so the result
    of this function will also vary with time.

    An hyperbola has (e>1 e.g. e=1.5)

    Args:
        tp: Time of perihelion passage in Julian centuries since J2000
        next_H_func : iterating function to calculate the eccentric anomaly, used
                for solving kepler equation with Newton 
        a : Semi-major axis of the orbit in [AU]
        e : eccentricity of the orbit (>1 for hiperbola
        t : time of the computation in Julian centuries since J2000
        
    Returns :
        Position (r) w.r.t. orbital plane in [AU] 
        Velocity (v) w.r.t orbital plane in [AU/days]

    """
    cte = sqrt(GM/a)
    # I have done the calculation and it is right
    # 2*pi/T == cte/a  so this is equivalent to the mean calculation done previously
    # Mean anomaly for the hyperbolic orbit
    Mh = cte*(t-tp)/a 
    # Initial Eccentric anomaly for the hyperbolic orbit depends on Mh
    if Mh >=0 :
        H0 = np.log(1.8+2*Mh/e)
    else :
        H0 = -np.log(1.8-2*Mh/e)
    # Solve the eccentric anomaly
    H = solve_ke_newton(e, next_H, Mh, H0)
    cosh_H = cosh(H)
    sinh_H = sinh(H)
    fac =  sqrt((e+1.0)*(e-1.0))
    rho = e*cosh_H - 1.0
    r = np.array([a*(e-cosh_H), a*fac*sinh_H, 0.0])
    v = np.array([-cte*sinh_H/rho, cte*fac*cosh_H/rho,0.0])
    return r,v


def calc_stumpff_values(E_2, epsilon=1e-7, max_iters=100):    
    """ 
    Computes the values for the Stumpff functions C1, C2, C3 

    Args:
        E_2: Square of eccentric anomaly in rads^2
        epsilon: relative accuracy 
        max_iters: 
        
    Returns :
        A tuple with the valus of c1, c2, c3 
    """
    c1, c2, c3 = 0.0, 0.0, 0.0
    to_add = 1.0
    for n in range(1 ,max_iters):
        c1 += to_add
        to_add /= (2.0*n)
        c2 += to_add
        to_add /= (2.0*n+1.0)
        c3 += to_add
        to_add *= -E_2
        if isclose(to_add, 0, abs_tol=epsilon) :
        #if np.isclose(to_add,epsilon,atol=epsilon):
            return c1, c2, c3
    logger.error(f"Not converged after {n} iterations")
    return c1, c2, c3



def parabolic_orbit (tp, q, e, t, max_iters=15):
    """ 
    Computes position (r) and velocity (v) vectors for hiperbolic orbits
    Pg 66 of Astronomy on the Personal computer
    The m_anomaly is the one that varies with time so the result
    of this function will also vary with time.

    An parabolic has (e=1)

    Args:
        tp: Time of perihelion passage in Julian centuries since J2000
        next_H_func : iterating function to calculate the eccentric anomaly, used
                for solving kepler equation with Newton 
        a : Semi-major axis of the orbit in [AU]
        e : eccentricity of the orbit (>1 for hiperbola
        t : time of the computation in Julian centuries since J2000
        
    Returns :
        Position (r) w.r.t. orbital plane in [AU] 
        Velocity (v) w.r.t orbital plane in [AU/days]

    """
    E_2 = 0.0    
    factor = 0.5 * e
    cte = sqrt(GM/(q*(1.0+e)))
    tau = sqrt(GM)*(t-tp)
    epsilon = 1e-7

    for i in range(max_iters):
        E20 = E_2 
        A = 1.5 * sqrt(factor/(q*q*q))*tau
        B = np.cbrt(sqrt(A*A+1.0)+A)
        u = B - 1.0/B 
        u_2 = u*u
        E_2 = u_2*(1.0-e)/factor 
        c1, c2, c3 = calc_stumpff_values(E_2)
        factor = 3.0*e*c3 
        if isclose(E_2, E20, abs_tol=epsilon) :
            R = q * (1.0 + u_2*c2*e/factor)
            r = np.array([q*(1.0-u_2*c2/factor), q*sqrt((1.0+e)/factor)*u*c1,0.0])
            v = np.array([-cte*r[1]/R, cte*(r[0]/R+e),0.0])
            return r,v
    #logger.warning(f"Not converged after {i} iterations")
    logger.error(f'Not converged with q:{q},  e:{e}, t:{t}, t0:{tp} after {i} iterations')
    return 0,0


M_min = 0.1

class KeplerianOrbit:

    def __init__(self, epoch_mjd,  q, a, e, tp_mjd, M_at_epoch) :
        self.epoch_mjd = epoch_mjd
        self.tp_mjd = tp_mjd
        self.q = q
        self.e = e
        self.a = a
        self.M_at_epoch = M_at_epoch        

    @classmethod
    def for_body(cls, body_elms):    
        return cls(body_elms.epoch_mjd, None, body_elms.a, body_elms.e, body_elms.tp_mjd, body_elms.M0)

    @classmethod
    def for_comet(cls, comet_elms):    
        return cls( comet_elms.epoch_mjd, comet_elms.q, None, comet_elms.e, comet_elms.tp_mjd, None)

    def calc_rv(self, t_mjd) :
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
        
        if self.a is not None :
            ## This is a body
            period_in_days = TWOPI*sqrt(pow(self.a,3)/GM)
            M = calc_M_for_body (t_mjd, self.epoch_mjd, period_in_days, self.M_at_epoch)
        else :
            # This is a comet
            # The inv_a is calculated to avoid to divide by 0 in parabolic
            inv_a = np.abs(1.0-self.e)/self.q
            M = calc_M_for_comet(t_mjd, self.tp_mjd, inv_a)
            # This is a comet

        if ((M < M_min) and (np.abs(1.0-self.e) < 0.1)) or isclose(self.e, 1.0, abs_tol=1e-04) :
            logger.warning(f'Doing parabolic orbit for e: {self.e}')
            xyz, vxyz = parabolic_orbit(self.tp_mjd, self.q, self.e, t_mjd, 50)
        elif self.e < 1.0 :
            a = self.q/(1.0-self.e) if self.a is None else self.a
            logger.warning(f'Doing elliptic orbit for e: {self.e}')
            xyz, vxyz = elliptic_orbit(next_E, M, a, self.e)
        else :
            logger.warning(f'Doing hyperbolic orbit for e: {self.e}')
            a = self.q/np.abs(1.0-self.e) if self.a is None else self.a
            xyz, vxyz =  hyperpolic_orbit (self.tp_mjd, next_H, a, self.e, t_mjd)
        return xyz, vxyz
      

def g_rlb_equat_body_j2000(jd, body):

    """
    Computes the geocentric polar coordinates of body based at particular time and orbital elements of hte 
    body following the second algorith of the Meeus (this is the one that explains that the 
    sum of the vectors are done in the equatorial system and it works best that doing in the eliptic system)

    Args:
        jd : Point in time (normally used in Julian days
        body: Orbital elements of the body

    Returns :
        A numpy vector of 3 positions with the geocentric coordinates of the body in Equatorial system
    """
   
    T_J2000 = 0.0 # desired equinox

    # The matrix will include the precesion from equinox of the date of the body
    mtx_equat_PQR =  multi_dot([co.mtx_equatFeclip(T_J2000),
                               co.mtx_eclip_prec(body.T_eqx0,T_J2000),
                               co.mtx_gauss_vectors(body.Node, body.i, body.w)                                                          
                               ])

	# The equatorial xyz of the sun referred to equinox J2000 for the 
	# specific moment
    g_xyz_equat_sun = g_xyz_equat_sun_j2000(jd)    
    
    # Fist calculation of the position of the body is done
    #M = calc_M(jd, body.tp_jd, body.a)
    M = calc_M_for_comet(jd, body.tp_jd, 1/body.a)
    xyz, _ = elliptic_orbit(next_E, M, body.a, body.e)
	# xyz are cartesians in the orbital plane (perifocal system) so we need to transform 
	# to equatorial
    		
    # This is key to work. The sum of the vectors is done in the equatorial system.
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    tau = np.linalg.norm(g_xyz_equat_body)*INV_C

    # Second try using tau
    #M = ob.calc_M(jd-tau, body.tp_jd, body.a)
    #M = calc_M(jd-tau, body.tp_jd, body.a)
    M = calc_M_for_comet(jd-tau, body.tp_jd, 1/body.a)

    xyz, _ = elliptic_orbit(next_E, M, body.a, body.e)
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)    
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    return co.polarFcartesian(g_xyz_equat_body)



    

if __name__ == "__main__" :
    None

 



    





    

