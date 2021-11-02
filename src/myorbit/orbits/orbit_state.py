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
from myorbit.util.timeut import hemisphere
import myorbit.data_catalog as dc

np.set_printoptions(precision=12)

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


class OrbitStateSolver(ABC):
    @classmethod
    def make(cls, tp_mjd, e, q, a, epoch, M_at_epoch):    
        if isclose(e, 1, abs_tol=1e-6):
            # Comets have q (distance to perihelion but asteroids do not have)
            if q is None :
                msg=f'A parabolic cannot be calculated because we dont have q (distance to perihelion)'
                print (msg)
                logger.error(msg)    
                return 
            else :
                msg=f'Doing parabolic orbit for e={e}, q={q} [AU]'
                print(msg)
                logger.info(msg)                  
                return ParabolicalStateSolver(tp_mjd, q)              
        elif 0<= e < 1 :
            if a is None:
                a = q / (1-e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a} [AU]')
            msg=f'Doing elliptical orbit tp={tp_mjd}, a={a} [AU], e={e}'
            print(msg)
            logger.warning(msg)            
            return EllipticalStateSolver(tp_mjd, a, e, epoch, M_at_epoch)
        else :
            if a is None:
                a = q / (1-e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a} [AU]')
            msg = f'Doing hyperbolical orbit for  a={a} [AU], e={e}'
            logger.warning(msg)
            print(msg)
            return HyperbolicalState(tp_mjd, a, e)

    def calc_rv (self, t_mjd):        
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

class EllipticalStateSolver(OrbitStateSolver) :
    """[summary]

    Parameters
    ----------
    OrbitState : [type]
        [description]
    """

    def __init__(self, tp_mjd, a, e, epoch_mjd, M_at_epoch):    
        self.tp_mjd = tp_mjd
        self.a = a
        self.e = e
        self.the_energy = -GM/(2*self.a)
        self.epoch_mjd = epoch_mjd
        self.M_at_epoch = M_at_epoch

    def calc_rv_basic (self, t_mjd):
        if (self.tp_mjd is None) or (self.tp_mjd == 0.0) :
            # For asteroids, there is no time at perihelion or distance to periheliion
            M = calc_M_for_body(t_mjd, self.epoch_mjd, self.a, self.M_at_epoch) 
        else :
            M = calc_M(t_mjd, self.tp_mjd, self.a)
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


class ParabolicalStateSolver(OrbitStateSolver) :

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

class HyperbolicalState(OrbitStateSolver) :
    
    def __init__(self, tp_mjd, a, e):    
        self.tp_mjd = tp_mjd
        self.a = a
        self.e = e
        self.the_energy = -GM/(2*self.a)

    def calc_rv_basic(self, t_mjd):
        return calc_rv_for_hyperbolic_orbit(self.tp_mjd, self.a, self.e, t_mjd)

    def energy(self):
        return self.the_energy
        
    def v(self, r) :
        return sqrt(2*(self.energy()+(GM/r)))

def test_elliptical():
    state = OrbitStateSolver.make(56198.22249000007, 0.99999074, 1.29609218, None, None, None)    
    T0_MJD = 56197.0
    for dt in range(0,10):
        t_mjd = T0_MJD + dt
        r_xyz, rdot_xyz, *other = state.calc_rv(t_mjd)
        print (f'r_xyz={r_xyz}, rdot_xyz={rdot_xyz}')

def test_hyperbolical():
    state = OrbitStateSolver.make(59311.54326000018, 1.06388423, 3.20746664, None, None, None)    
    T0_MJD = 56197.0
    for dt in range(0,10):
        t_mjd = T0_MJD + dt
        r_xyz, rdot_xyz, *other = state.calc_rv(t_mjd)
        print (f'r_xyz={r_xyz}, rdot_xyz={rdot_xyz}')


def test_parabolical():
    state = OrbitStateSolver.make(57980.231000000145, 1.0, 2.48315593, None, None, None)    
    T0_MJD = 56197.0
    for dt in range(0,10):
        t_mjd = T0_MJD + dt
        r_xyz, rdot_xyz, *other = state.calc_rv(t_mjd)
        print (f'r_xyz={r_xyz}, rdot_xyz={rdot_xyz}')


if __name__ == "__main__" :
    test_elliptical()
    test_hyperbolical()
    test_parabolical()


