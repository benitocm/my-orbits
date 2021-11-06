"""This module contains functions related to the calculation of Keplerian orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
"""
# Standard library imports
from math import isclose
import logging
from abc import ABC, abstractmethod

# Third party imports
import numpy as np
from numpy import sqrt


# Local application imports
from myorbit.orbits.parabolic import calc_rv_for_parabolic_orbit
from myorbit.orbits.hyperbolic import calc_rv_for_hyperbolic_orbit
from myorbit.orbits.ellipitical import calc_rv_for_elliptic_orbit, calc_M_for_body, calc_M
from myorbit.orbits.near_parabolic import calc_rv_by_stumpff
from myorbit.util.timeut import hemisphere, mjd2str_date
from myorbit.util.general import  NoConvergenceError

from myorbit.util.constants import *


logger = logging.getLogger(__name__)

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

    num = E - e * np.sin(E) - m_anomaly
    den = 1 - e * np.cos(E)
    return E - num/den

#See https://www.giacomodebidda.com/posts/factory-method-and-abstract-factory-in-python/

class KeplerianStateSolver(ABC):
    @classmethod
    def make(cls, e, a=None, tp_mjd=None, q=None, epoch=None, M_at_epoch=None):    
        """[summary]

        Parameters
        ----------
        e : [type]
            [description]
        a : [type], optional
            [description], by default None
        tp_mjd : [type], optional
            [description], by default None
        q : [type], optional
            [description], by default None
        epoch : [type], optional
            [description], by default None
        M_at_epoch : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        
        if isclose(e, 1, abs_tol=1e-6):
            # Comets have q (distance to perihelion but asteroids do not have)
            if q is None :
                msg=f'A parabolic orbit cannot be calculated because q (distance to perihelion) is unknown'
                print (msg)
                logger.error(msg)    
                return None
            else :
                msg=f'Doing parabolic orbit for tp={tp_mjd}, e={e}, q={q} [AU]'
                print(msg)
                logger.info(msg)                  
                return ParabolicalStateSolver(tp_mjd=tp_mjd, q=q, e=e)              
        elif 0<= e < 1 :
            if a is None:
                a = q / (1-e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a} [AU]')
            msg=f'Doing elliptical orbit tp={tp_mjd}, a={a} [AU], e={e}'
            print(msg)
            logger.info(msg)            
            return EllipticalStateSolver(q=q, tp_mjd= tp_mjd, a=a, e=e, epoch_mjd = epoch, M_at_epoch=M_at_epoch)
        else :
            if a is None:
                a = q / (1-e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a} [AU]')
            msg = f'Doing hyperbolical orbit for tp={tp_mjd}, a={a} [AU], e={e}'
            logger.info(msg)
            print(msg)
            return HyperbolicalState(tp_mjd=tp_mjd, a=a, e=e)

    def calc_rv (self, t_mjd): 
        """ Template method pattern, that will call the concrete method calc_rv_basic in
        the specific subclass (depeding on the orbit type) and after that
        it will do the velocity and angular momentum checks that should hold on each t

        Parameters
        ----------
        t_mjd : float
            Time of computation

        Returns
        -------
        Tuple 
            (r_xyz, rdot_xyz, r, h) where

        """         
        r_xyz, rdot_xyz, r, h_xyz, *others = self.calc_rv_basic (t_mjd)    
        check_velocity(self.v(r), rdot_xyz)
        check_angular_momentum(np.linalg.norm(h_xyz), r_xyz, rdot_xyz)
        e_xyz = calc_eccentricity_vector(r_xyz, rdot_xyz, h_xyz)
        e = np.linalg.norm(e_xyz)
        if not isclose(e, self.e, rel_tol=0, abs_tol=1e-05):
            msg=f'The modulus of the eccentricity vector {e} is not equal to the eccentrity {self.e}'
            print (msg)
            logger.warning(msg)

        return r_xyz, rdot_xyz, r, h_xyz, e_xyz

    @abstractmethod
    def calc_rv_basic(self, t_mjd):
        pass

    @abstractmethod
    def energy(self):
        pass
        
    @abstractmethod
    def v(self, r) :
        pass
                
def check_velocity(v, rdot_xyz):
    if not isclose(v,np.linalg.norm(rdot_xyz),abs_tol=1e-12):
        msg=f'The velocity does not match,  v_energy={v}, modulus of rdot_xyz={np.linalg.norm(rdot_xyz)}'
        print(msg)
        logger.error(msg)

def check_angular_momentum(h, r_xyz, rdot_xyz):
        h_rv = np.cross(r_xyz,rdot_xyz)
        if not isclose(h_rv[2], h, abs_tol=1e-12):
            msg=f'The angular momentum does not match h_rv={h_rv[2]}, h_geometric={h}'
            print(msg)
            logger.error(msg)


def calc_eccentricity_vector(r_xyz, rdot_xyz, h):
    return  (np.cross(rdot_xyz,h) - (GM*r_xyz/np.linalg.norm(r_xyz)))/GM



# There are a set of comets with an eccentricity like 0.999986, 0.999914 (elliptical orbits)
# that near the perihelion passage, i.e., Mean Anomaly close to TWOPI or 0, the calculation of
# Eccentric anomaly does not converge. In those specific points, the near parabolic will be used.

class EllipticalStateSolver(KeplerianStateSolver) :
    """[summary]

    Parameters
    ----------
    KeplerianStateSolver : [type]
        [description]
    """

    def __init__(self, a, e, q=None, tp_mjd=None, epoch_mjd=None, M_at_epoch=None):    
        self.a = a
        self.e = e
        self.q = q
        self.tp_mjd = tp_mjd
        # The energy is an invariant of the orbit. In this case, it is negative
        self.the_energy = - GM/(2*self.a)
        self.epoch_mjd = epoch_mjd
        self.M_at_epoch = M_at_epoch

    def calc_rv_basic (self, t_mjd):
        if self.tp_mjd is None:
            # For asteroids, there is no time at perihelion or distance to periheliion
            M = calc_M_for_body(t_mjd=t_mjd, epoch_mjd= self.epoch_mjd, a= self.a, M_at_epoch= self.M_at_epoch) 
        else :
            # For comets, time at perihelion and distance to perihelion is known
            M = calc_M(t_mjd=t_mjd, tp_mjd=self.tp_mjd, a=self.a)
        try :
            r_xyz, rdot_xyz, r, h_xyz, M, f, E = calc_rv_for_elliptic_orbit (M, self.a, self.e)
            if hemisphere(f) != hemisphere(M):
                msg=f'The hemisphere of True anomaly {np.rad2deg(f)} degress {hemisphere(f)} is different from the hemisphere of Mean anomaly {np.rad2deg(M)} {hemisphere(M)}'
                print(msg)
                logger.error(msg)
            return r_xyz, rdot_xyz, r, h_xyz, M, f, E
        except NoConvergenceError as ex:
            msg = f'NOT converged, for M={M} at time={mjd2str_date(t_mjd)} with root={ex.root}'
            print (msg)
            logger.error(msg)
            msg = f'Trying with the near parabolical method with tp={self.tp_mjd}, q={self.q} AU, e={self.e} at time {mjd2str_date(t_mjd)} '
            print (msg)
            logger.error(msg)
            r_xyz, rdot_xyz, r, h_xyz =  calc_rv_by_stumpff (self.tp_mjd, self.q, self.e, t_mjd)
            return r_xyz, rdot_xyz, r, h_xyz, -100, -100, -100

    def energy(self):
        return self.the_energy
                
    def v(self, r) :
         return sqrt(2*(self.energy()+(GM/r)))


class ParabolicalStateSolver(KeplerianStateSolver) :
    def __init__(self, tp_mjd, q, e):    
        self.tp_mjd = tp_mjd
        self.q = q
        self.e = e
        # The energy is an invariant of the orbit. In this case, it is 0
        self.the_energy = 0

    def calc_rv_basic(self, t_mjd):
        return calc_rv_for_parabolic_orbit (tp_mjd= self.tp_mjd, q=self.q, t_mjd= t_mjd)          

    def energy(self):
        return self.the_energy
        
    def v(self, r) :
        return np.sqrt(2*GM/r)

class HyperbolicalState(KeplerianStateSolver) :
    
    def __init__(self, tp_mjd, a, e):    
        self.tp_mjd = tp_mjd
        self.a = a
        self.e = e
        # The energy is an invariant of the orbit. In this case, it is < 0 
        self.the_energy = -GM/(2*self.a)

    def calc_rv_basic(self, t_mjd):
        return calc_rv_for_hyperbolic_orbit(tp_mjd= self.tp_mjd, a_neg=self.a, e=self.e, t_mjd=t_mjd)

    def energy(self):
        return self.the_energy
        
    def v(self, r) :
        return sqrt(2*(self.energy()+(GM/r)))

if __name__ == "__main__" :
    None

