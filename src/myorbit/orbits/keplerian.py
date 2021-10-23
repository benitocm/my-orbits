"""This module contains functions related to the calculation of Keplerian orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
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


def _calc_M_for_body(t_mjd, epoch_mjd, period_in_days, M_at_epoch) :

    """Computes the mean anomaly based on the data of BodyElms, in this case,
    uses the period (calculated) and the Mean anomaly at epoch.

    Parameters
    ----------
    t_mjd : float
        Time of the computation as Modified Julian Day
    epoch_mjd : float
        The Epoch considered
    period_in_days : float
         period of the orbit [days]
    M_at_epoch : float
        The mean anomaly at epoch [radians]

    Returns
    -------
    float
        The mean anomaly [radians]
    """
    M = (t_mjd - epoch_mjd)*TWOPI/period_in_days
    M += M_at_epoch
    return reduce_rad(M,to_positive=True)     


def _calc_M_for_comet(t_mjd, tp_mjd, inv_a) :

    """Computes the mean anomaly as a function fo t, t0 and a, i.e., not depending on the
    period of the orbit) and semi-major axis

    Parameters
    ----------
    t_mjd : float
        Time of the computation as Modified Julian Day
    tp_mjd : float
        Time of perihelion passage 
    inv_a : float
        Inverse of the semi-major axis [AU]

    Returns
    -------
    float
        The mean anomaly [radians]
    """

    #M = sqrt(ut.GM)*(t_mjd-tp_mjd)/np.float_power(a, 1.5)
    M = sqrt(GM)*(t_mjd-tp_mjd)*sqrt(pow(inv_a, 3))
    return reduce_rad(M, to_positive=True)


def _calc_M_by_E(e, E):
    """Computes the true anomaly based on the eccentricity value and eccentric anomaly

    Parameters
    ----------
    e : float
        Eccentricity of the orbit
    E : float
        The eccentric anomaly [radians]

    Returns
    -------
    float
        The true anomaly [radians]
    """

    return 2 * arctan(sqrt((1+e)/(1-e))* tan(E/2))    

def _next_E (e, m_anomaly, E) :
    """Computes the eccentric anomaly for elliptical orbits. This 
    function will be called by an iterative procedure. Used in the Newton
    method (Pag 65 of Astronomy of  the personal computer book)

    Parameters
    ----------
    e : float
        Eccentricity of the orbit
    m_anomaly : float
        Mean anomaly [radians]
    E : float
        The eccentric anomaly (radians)

    Returns
    -------
    float
        The next value of the the eccentric anomaly [radians]
    """

    num = E - e * sin(E) - m_anomaly
    den = 1 - e * cos(E)
    return E - num/den

def _elliptic_orbit (next_E_func, m_anomaly, a, e):
    """Computes the position (r) and velocity (v) vectors for elliptic orbits using
    an iterative method (Newton's method) for solving the Kepler equation.
    (pg 66 of Astronomy on the Personal Computer book). The m_anomaly is
    the one that varies with time so the result of this function will also vary with time.

    Parameters
    ----------
    next_E_func : function
        Function that really calculates the eccentric anomaly used in the iterative 
        solution procedure (Newton's method for solving the Kepler equation)
    m_anomaly : float
        The current Mean anomaly that will depend on time [rads]
    a : float
        Semi-major axis of the orbit [AU]
    e : float
        Eccentricity of the orbit (<1 for elliptical)

    Returns
    -------
    tuple (r,v):
        Where r is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane [AU]
        Where v is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane [AU/days]
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

def _next_H (e, mh_anomaly, H):
    """Computes the eccentric anomaly for hyperbolic orbits. This 
    function will be called by an iterative procedure. Used in the Newton
    method (Pag 65 of Astronomy of  the personal computer book)

    Parameters
    ----------
    e : float
        eccentricity of the orbit
    mh_anomaly : float
        Mean anomaly in angle units [radians]
    H : float
        Previous value of the eccentric anomaly [radians]

    Returns
    -------
    float
        The next value of The eccentric anomaly [radians]
    """
    num = e * sinh(H) - H - mh_anomaly
    den = e * cosh(H) - 1
    return H - num/den


def _hyperpolic_orbit (tp, next_H_func, a, e, t):
    """Computes the position (r) and velocity (v) vectors for hyperbolic orbits using
    an iterative method (Newton's method) for solving the Kepler equation.
    (pg 66 of Astronomy on the Personal Computer book). The m_anomaly is
    the one that varies with time so the result of this function will also vary with time.

    Parameters
    ----------
    tp : float
        Time of perihelion passage in Julian centuries since J2000
    next_H_func : function
        Function that really calculates the eccentric anomaly used in the iterative 
        solution procedure (Newton's method for solving the Kepler equation)
    a : float
        Semi-major axis of the orbit in [AU]
    e : float
        Eccentricity of the orbit (>1 for hiperbola)
    t : float
        Time of the computation in Julian centuries since J2000

    Returns
    -------
    tuple (r,v):
        Where r is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane [AU]
        Where v is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane [AU/days]
            
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
    H = solve_ke_newton(e, next_H_func, Mh, H0)
    cosh_H = cosh(H)
    sinh_H = sinh(H)
    fac =  sqrt((e+1.0)*(e-1.0))
    rho = e*cosh_H - 1.0
    r = np.array([a*(e-cosh_H), a*fac*sinh_H, 0.0])
    v = np.array([-cte*sinh_H/rho, cte*fac*cosh_H/rho,0.0])
    return r,v


def _calc_stumpff_values(E_2, epsilon=1e-7, max_iters=100):    
    """Computes the values for the Stumpff functions C1, C2, C3 following
    an iterative procedure

    Parameters
    ----------
    E_2 : float
        Square of eccentric anomaly [radians^2]
    epsilon : float, optional
        Relative accuracy, by default 1e-7
    max_iters : int, optional
        Maximum number of iterations, by default 100

    Returns
    -------
    tuple (C1, C2, C3)
        The three stumpff values [float]
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

def _parabolic_orbit (tp, q, e, t, max_iters=15):
    """Computes the position (r) and velocity (v) vectors for parabolic orbits using
    an iterative method (Newton's method) for solving the Kepler equation.
    (pg 66 of Astronomy on the Personal Computer book). The m_anomaly is
    the one that varies with time so the result of this function will also vary with time.

    Parameters
    ----------
    tp : float
        Time of perihelion passage in Julian centuries since J2000
    q : float
        Perihelion distance [AU]
    e : float
        Eccentricity of the orbit
    t : float
        Time of the computation in Julian centuries since J2000
    max_iters : int, optional
        Maximum number of iterations, by default 15

    Returns
    -------
    tuple (r,v):
        Where r is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
            with respect to the orbital plane [AU]
        Where v is a np.array[3] that contains the velocity vector (cartesian) of the body
            with respect to the orbital plane [AU/days]
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
        c1, c2, c3 = _calc_stumpff_values(E_2)
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
        """Setup the parameters for the orbit calculation

        Parameters
        ----------
        epoch_mjd : [type]
            [description]
        q : float
            Perihelion distance [AU]
        a : float
            [description]
        e : float
            Eccentricity of the orbit
        tp_mjd : float
            Time reference at which the object passed through perihelion in Modified Julian Day
        M_at_epoch : float
            Mean anomaly at epoch
        """
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
        """Computes position (r) and velocity (v) vectors for keplerian orbits
        depending the eccentricy and mean_anomlay to choose which type of conic use

        Parameters
        ----------
        t_mjd : float
            Time of the computation in Julian centuries since J2000

        Returns
        -------
        tuple (r,v):
            Where r is a np.array[3] that contains the radio vector (cartesian) from the Sun to the body 
                with respect to the orbital plane [AU]
            Where v is a np.array[3] that contains the velocity vector (cartesian) of the body
                with respect to the orbital plane [AU/days]
        """
        
        if self.a is not None :
            ## This is a body
            period_in_days = TWOPI*sqrt(pow(self.a,3)/GM)
            M = _calc_M_for_body (t_mjd, self.epoch_mjd, period_in_days, self.M_at_epoch)
        else :
            # This is a comet
            # The inv_a is calculated to avoid to divide by 0 in parabolic
            inv_a = np.abs(1.0-self.e)/self.q
            M = _calc_M_for_comet(t_mjd, self.tp_mjd, inv_a)

        if ((M < M_min) and (np.abs(1.0-self.e) < 0.1)): #or isclose(self.e, 1.0, abs_tol=1e-04) :
            #logger.warning(f'Doing parabolic orbit for e: {self.e} and M: {M}')
            print(f'Doing parabolic orbit for e: {self.e} and M: {M}')
            xyz, vxyz = _parabolic_orbit(self.tp_mjd, self.q, self.e, t_mjd, 50)
        elif self.e < 1.0 :
            a = self.q/(1.0-self.e) if self.a is None else self.a
            #logger.warning(f'Doing elliptic orbit for e: {self.e} and M: {M}')
            print(f'Doing elliptic orbit for e: {self.e} and M: {M}')
            xyz, vxyz = _elliptic_orbit(_next_E, M, a, self.e)
        else :
            #logger.warning(f'Doing hyperbolic orbit for e: {self.e} and M: {M}')
            print(f'Doing hyperbolic orbit for e: {self.e} and M: {M}')
            a = self.q/np.abs(1.0-self.e) if self.a is None else self.a
            xyz, vxyz =  _hyperpolic_orbit (self.tp_mjd, _next_H, a, self.e, t_mjd)
        return xyz, vxyz
      
#
# Currently this code is not being used but I want to keep it because the same idea is used 
# in othe parts, the sum of vectors are don in the equatorial system rather than the ecliptic system
# 

def _g_rlb_equat_body_j2000(jd, body):    
    """Computes the geocentric polar coordinates of body (eliptical orbit) based at particular time and orbital elements of hte 
    body following the second algorithm of the Meeus (this is the one that explains that the 
    sum of the vectors are done in the equatorial system and it works best that doing in the eliptic system)

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
    mtx_equat_PQR =  multi_dot([co.mtx_equatFeclip(T_J2000),
                               co.mtx_eclip_prec(body.T_eqx0,T_J2000),
                               co.mtx_gauss_vectors(body.Node, body.i, body.w)                                                          
                               ])

	# The equatorial xyz of the sun referred to equinox J2000 for the 
	# specific moment
    g_xyz_equat_sun = g_xyz_equat_sun_j2000(jd)    
    
    # Fist calculation of the position of the body is done
    #M = calc_M(jd, body.tp_jd, body.a)
    M = _calc_M_for_comet(jd, body.tp_jd, 1/body.a)
    xyz, _ = _elliptic_orbit(_next_E, M, body.a, body.e)
	# xyz are cartesians in the orbital plane (perifocal system) so we need to transform 
	# to equatorial
    		
    # This is key to work. The sum of the vectors is done in the equatorial system.
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    tau = np.linalg.norm(g_xyz_equat_body)*INV_C

    # Second try using tau
    #M = ob.calc_M(jd-tau, body.tp_jd, body.a)
    #M = calc_M(jd-tau, body.tp_jd, body.a)
    M = _calc_M_for_comet(jd-tau, body.tp_jd, 1/body.a)

    xyz, _ = _elliptic_orbit(_next_E, M, body.a, body.e)
    h_xyz_equat_body = mtx_equat_PQR.dot(xyz)    
    g_xyz_equat_body = g_xyz_equat_sun + h_xyz_equat_body
    return co.polarFcartesian(g_xyz_equat_body)

    

if __name__ == "__main__" :
    None

 



    





    

