"""This module contains functions related to the calculation of Keplerian orbits.
It is based on the book "Astronomy on the Personal Computer" by Montenbruck, Pfleger.
"""
# Standard library imports
from math import isclose
import logging
from collections import namedtuple



# Third party imports
import numpy as np
from numpy import sqrt, cos, sin, cosh, sinh, tan, arctan
from numpy.linalg import multi_dot
from scipy.optimize import newton

# Local application imports
from myorbit.util.timeut import  norm_rad
from myorbit.orbits.orbutil import solve_ke_newton
from myorbit.planets import g_xyz_equat_sun_j2000
from myorbit.util.general import pow
from myorbit import coord as co
from myorbit.orbits.parabolic import calc_rv_for_parabolic_orbit
from myorbit.orbits.hyperbolic import calc_rv_for_hyperbolic_orbit
from myorbit.orbits.ellipitical import calc_rv_for_elliptic_orbit
import myorbit.data_catalog as dc

np.set_printoptions(precision=12)

from myorbit.util.constants import *

logger = logging.getLogger(__name__)


State = namedtuple("State", "r_xyz, rdot_xyz")

class KeplerianOrbit:

    def __init__(self, epoch_mjd, q, a, e, tp_mjd, M_at_epoch) :
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
        orbit_type = None
        if isclose(self.e, 1, abs_tol=1e-6):
            # Comets have q (distance to perihelion but bodies do not have)
            if self.q is None :
                msg=f'An parabolic cannot be calculated because we dont have q (distance to perihelion)'
                print (msg)
                logger.error(msg)    
                return None,None
            else :
                msg=f'Doing parabolic orbint for t={t_mjd}, tp={self.tp_mjd}, q={self.q}'
                print(msg)
                logger.info(msg)                
                orbit_type = None
                return calc_rv_for_parabolic_orbit(t_mjd, self.tp_mjd, self.q)
        elif 0<= self.e < 1 :
            logger.warning(f'Doing elliptical orbit for e: {self.e}')
            print(f'Doing elliptical orbit for e: {self.e}')
            return np.array[1,1,1], np.array[1,1,1]
        else :
            if self.a is None:
                a = self.q / (1-self.e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a}')
            else :
                a = self.a
            msg = f'Doing hyperbolical orbit for t={t_mjd}, tp={self.tp_mjd}, a={a}, e={self.e}'
            logger.warning(msg)
            print(msg)
            return calc_rv_for_hyperbolic_orbit(t_mjd, self.tp_mjd, a, self.e)

    def calc_orb(self, t0_mjd, step, tf_mjd) :
        """
        orbit_type = None
        if isclose(self.e, 1, abs_tol=1e-6):
            orbit_type = 'parabolic'
            # Comets have q (distance to perihelion but bodies do not have)
            if self.q is None :
                msg=f'An parabolic cannot be calculated because we dont have q (distance to perihelion)'
                print (msg)
                logger.error(msg)    
                return 
            else :
                msg=f'Doing parabolic orbit for tp={self.tp_mjd}, q={self.q}'
                print(msg)
                logger.info(msg)                                
        elif 0<= self.e < 1 :
            orbit_type = 'elliptical'
            if self.a is None:
                a = self.q / (1-self.e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a}')
            else :
                a = self.a            
            msg=f'Doing elliptical orbit tp={self.tp_mjd}, a={a}, e={self.e}'
            print(msg)
            logger.warning(msg)            
        else :
            orbit_type = 'hyperbolic'
            if self.a is None:
                a = self.q / (1-self.e) 
                print (f'Semi-major axis (a) not provided, calculated with value {a}')
            else :
                a = self.a
            msg = f'Doing hyperbolical orbit for tp={self.tp_mjd}, a={a}, e={self.e}'
            logger.warning(msg)
            print(msg)
        """
        t_mjd = t0_mjd 
        hs= []
        state_list = [] 
        while True :
            if orbit_type == 'parabolic':
                # Parabolic orbit
                r_xyz, rdot_xyz, r, h_geo,  *others =  calc_rv_for_parabolic_orbit (self.tp_mjd, self.q, t_mjd)
                # In this case, Energy = 0
                v = np.sqrt(2*GM/r)
            elif orbit_type == 'elliptical':
                # Elliptical
                r_xyz, rdot_xyz, r, h_geo, *others =  calc_rv_for_elliptic_orbit (self.tp_mjd, a, self.e, t_mjd)
                Energy = - GM/(2*a)
                v = sqrt(2*(Energy+(GM/r)))
            else : 
                # Hyperbolic
                r_xyz, rdot_xyz, r, h_geo, *others = calc_rv_for_hyperbolic_orbit (self.tp_mjd, a, self.e, t_mjd)
                Energy = - GM/(2*a)
                v = sqrt(2*(Energy+(GM/r)))
            # For every orbit and state calculated, the velocity double check is done
            if not isclose(v,np.linalg.norm(rdot_xyz),abs_tol=1e-12):
                msg=f'The velocity does not match v_energy={v}, modulus of rdot_xyz={np.linalg.norm(rdot_xyz)}'
                print(msg)
                logger.error(msg)
            # For every orbit and state calculated, the angular momentum check is done
            h_rv = np.cross(r_xyz,rdot_xyz)
            if not isclose(h_rv[2], h_geo, abs_tol=1e-12):
                msg=f'The angular momentum does not match h_rv={h_rv[2]}, h_geometric={h_geo}'
                print(msg)
                logger.error(msg)
            hs.append(h_geo)         
            state_list.append(State(r_xyz=r_xyz, rdot_xyz=rdot_xyz))   
            t_mjd = t_mjd + step
            if t_mjd > tf_mjd:
                break
        if not all(isclose(h, hs[0], abs_tol=1e-12) for h in hs):
            msg=f'The angular momentum is not constant'
            print(msg)
            logger.error(msg)
        return state_list

      
def test_parabolic():
    k_orbit = KeplerianOrbit.for_comet(dc.C_2018_F3_Johnson)
    T0_MJD = 57966.0
    state_list = k_orbit.calc_orb(T0_MJD, 2, T0_MJD+100)
    print (state_list)
    

def test_hyperbolic():
    T0_MJD = 59205.0
    k_orbit = KeplerianOrbit.for_comet(dc.C_2020_J1_SONEAR)
    k_orbit.calc_orb(T0_MJD, 2, T0_MJD+100)

def test_elliptical():
    T0_MJD = 56197.0
    k_orbit = KeplerianOrbit.for_comet(dc.C2012_CH17)
    k_orbit.calc_orb(T0_MJD, 2, T0_MJD+100)



if __name__ == "__main__" :
    test_parabolic()
    #test_hyperbolic()
    #test_elliptical()

 



    





    

