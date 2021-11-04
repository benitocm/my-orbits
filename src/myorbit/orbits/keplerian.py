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
from myorbit.orbits.hyperbolic import calc_rv_for_hyperbolic_orbit, _parabolic_orbit
from myorbit.orbits.ellipitical import calc_rv_for_elliptic_orbit, calc_M_for_body, calc_M
from myorbit.util.timeut import hemisphere

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
    def make(cls, tp_mjd, e, q, a, epoch, M_at_epoch):    
        """[summary]

        Parameters
        ----------
        tp_mjd : [type]
            [description]
        e : [type]
            [description]
        q : [type]
            [description]
        a : [type]
            [description]
        epoch : [type]
            [description]
        M_at_epoch : [type]
            [description]

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
                return 
            else :
                msg=f'Doing parabolic orbit for tp={tp_mjd}, e={e}, q={q} [AU]'
                print(msg)
                logger.info(msg)                  
                return ParabolicalStateSolver(tp_mjd=tp_mjd, q=q)              
        elif 0<= e < 1 :
            if a is None:
                a = q / (1-e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a} [AU]')
            msg=f'Doing elliptical orbit tp={tp_mjd}, a={a} [AU], e={e}'
            print(msg)
            logger.warning(msg)            
            return EllipticalStateSolver(tp_mjd= tp_mjd, a=a, e=e, epoch_mjd = epoch, M_at_epoch=M_at_epoch)
        else :
            if a is None:
                a = q / (1-e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a} [AU]')
            msg = f'Doing hyperbolical orbit for tp={tp_mjd}, a={a} [AU], e={e}'
            logger.warning(msg)
            print(msg)
            return HyperbolicalState(tp_mjd=tp_mjd, a=a, e=e, q=q)

    def calc_rv (self, t_mjd): 
        """ Template method pattern, that will call the concrete method calc_rv_basic in
        the specific subclass (depeding on the orbit type) and after that
        it will do the velocity and angular momentum checks that should hold

        Parameters
        ----------
        t_mjd : float
            Time of computation

        Returns
        -------
        Tuple 
            (r_xyz, rdot_xyz, r, h) where

        """         
        r_xyz, rdot_xyz, r, h, *others = self.calc_rv_basic (t_mjd)
        check_velocity(self.v(r), rdot_xyz)
        check_angular_momentum(h, r_xyz, rdot_xyz)
        return r_xyz, rdot_xyz, r, h

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
        msg=f'The velocity does not match v_energy={v}, modulus of rdot_xyz={np.linalg.norm(rdot_xyz)}'
        print(msg)
        logger.error(msg)

def check_angular_momentum(h, r_xyz, rdot_xyz):
        h_rv = np.cross(r_xyz,rdot_xyz)
        if not isclose(h_rv[2], h, abs_tol=1e-12):
            msg=f'The angular momentum does not match h_rv={h_rv[2]}, h_geometric={h}'
            print(msg)
            logger.error(msg)

class EllipticalStateSolver(KeplerianStateSolver) :
    """[summary]

    Parameters
    ----------
    KeplerianStateSolver : [type]
        [description]
    """

    def __init__(self, tp_mjd, a, e, epoch_mjd, M_at_epoch):    
        self.tp_mjd = tp_mjd
        self.a = a
        self.e = e
        # The energy is a invariant of the orbit 
        self.the_energy = -GM/(2*self.a)
        self.epoch_mjd = epoch_mjd
        self.M_at_epoch = M_at_epoch

    def calc_rv_basic (self, t_mjd):
        if (self.tp_mjd is None) or (self.tp_mjd == 0.0) :
            # For asteroids, there is no time at perihelion or distance to periheliion
            M = calc_M_for_body(t_mjd=t_mjd, epoch_mjd= self.epoch_mjd, a= self.a, M_at_epoch= self.M_at_epoch) 
        else :
            M = calc_M(t_mjd=t_mjd, tp_mjd=self.tp_mjd, a=self.a)
        r_xyz, rdot_xyz, r, h, M, f, E = calc_rv_for_elliptic_orbit (M, self.a, self.e)
        if hemisphere(f) != hemisphere(M):
            msg=f'The hemisphere of True anomaly {np.rad2deg(f)} degress {hemisphere(f)} is different from the hemisphere of Mean anomaly {np.rad2deg(M)} {hemisphere(M)}'
            print(msg)
            logger.error(msg)
        return r_xyz, rdot_xyz, r, h, M, f, E

    def energy(self):
        return self.the_energy
                
    def v(self, r) :
         return sqrt(2*(self.energy()+(GM/r)))


class ParabolicalStateSolver(KeplerianStateSolver) :
    def __init__(self, tp_mjd, q):    
        self.tp_mjd = tp_mjd
        self.q = q
        self.the_energy = 0

    def calc_rv_basic(self, t_mjd):
        return calc_rv_for_parabolic_orbit (self.tp_mjd, self.q, t_mjd)

    def energy(self):
        return self.the_energy
        
    def v(self, r) :
        return np.sqrt(2*GM/r)

class HyperbolicalState(KeplerianStateSolver) :
    
    def __init__(self, tp_mjd, a, e, q):    
        self.tp_mjd = tp_mjd
        self.a = a
        self.e = e
        self.q = q
        self.the_energy = -GM/(2*self.a)

    def calc_rv_basic(self, t_mjd):
        return calc_rv_for_hyperbolic_orbit(self.tp_mjd, self.a, self.e, t_mjd)
        #return _parabolic_orbit (self.tp_mjd, self.q, self.e, t_mjd, max_iters=15)


    def energy(self):
        return self.the_energy
        
    def v(self, r) :
        return sqrt(2*(self.energy()+(GM/r)))


if __name__ == "__main__" :
    None

