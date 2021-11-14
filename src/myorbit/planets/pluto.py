"""This module contains functions to obtain the position Pluto
"""

# Standard library imports

# Third party imports
import numpy as np
from numpy import deg2rad
from toolz import pipe 

# Local application imports
from myorbit.util.timeut import JD_J2000, CENTURY
from myorbit.util.general import pr_radv, kahan_sum
from myorbit.coord import cartesianFpolar, obliquity, Rx_3d, polarFcartesian
from .vsop87 import g_xyz_equat_sun_j2000

"""This table contains Pluto's argument coefficients according to Table 37.A in
Meeus' book, page 265.
"""
ARGUMENT = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 2.0],
    [0.0, 0.0, 3.0],
    [0.0, 0.0, 4.0],
    [0.0, 0.0, 5.0],
    [0.0, 0.0, 6.0],
    [0.0, 1.0, -1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.0, 1.0, 2.0],
    [0.0, 1.0, 3.0],
    [0.0, 2.0, -2.0],
    [0.0, 2.0, -1.0],
    [0.0, 2.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, -1.0, 1.0],
    [1.0, 0.0, -3.0],
    [1.0, 0.0, -2.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 2.0],
    [1.0, 0.0, 3.0],
    [1.0, 0.0, 4.0],
    [1.0, 1.0, -3.0],
    [1.0, 1.0, -2.0],
    [1.0, 1.0, -1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 3.0],
    [2.0, 0.0, -6.0],
    [2.0, 0.0, -5.0],
    [2.0, 0.0, -4.0],
    [2.0, 0.0, -3.0],
    [2.0, 0.0, -2.0],
    [2.0, 0.0, -1.0],
    [2.0, 0.0, 0.0],
    [2.0, 0.0, 1.0],
    [2.0, 0.0, 2.0],
    [2.0, 0.0, 3.0],
    [3.0, 0.0, -2.0],
    [3.0, 0.0, -1.0],
    [3.0, 0.0, 0.0]
])

LONGITUDE = np.array([
    [-19799805.0, 19850055.0],
    [897144.0, -4954829.0],
    [611149.0, 1211027.0],
    [-341243.0, -189585.0],
    [129287.0, -34992.0],
    [-38164.0, 30893.0],
    [20442.0, -9987.0],
    [-4063.0, -5071.0],
    [-6016.0, -3336.0],
    [-3956.0, 3039.0],
    [-667.0, 3572.0],
    [1276.0, 501.0],
    [1152.0, -917.0],
    [630.0, -1277.0],
    [2571.0, -459.0],
    [899.0, -1449.0],
    [-1016.0, 1043.0],
    [-2343.0, -1012.0],
    [7042.0, 788.0],
    [1199.0, -338.0],
    [418.0, -67.0],
    [120.0, -274.0],
    [-60.0, -159.0],
    [-82.0, -29.0],
    [-36.0, -29.0],
    [-40.0, 7.0],
    [-14.0, 22.0],
    [4.0, 13.0],
    [5.0, 2.0],
    [-1.0, 0.0],
    [2.0, 0.0],
    [-4.0, 5.0],
    [4.0, -7.0],
    [14.0, 24.0],
    [-49.0, -34.0],
    [163.0, -48.0],
    [9.0, -24.0],
    [-4.0, 1.0],
    [-3.0, 1.0],
    [1.0, 3.0],
    [-3.0, -1.0],
    [5.0, -3.0],
    [0.0, 0.0]
])

"""This table contains the periodic terms to compute Pluto's heliocentric
longitude according to Table 37.A in Meeus' book, page 265"""

LATITUDE = np.array([
    [-5452852.0, -14974862],
    [3527812.0, 1672790.0],
    [-1050748.0, 327647.0],
    [178690.0, -292153.0],
    [18650.0, 100340.0],
    [-30697.0, -25823.0],
    [4878.0, 11248.0],
    [226.0, -64.0],
    [2030.0, -836.0],
    [69.0, -604.0],
    [-247.0, -567.0],
    [-57.0, 1.0],
    [-122.0, 175.0],
    [-49.0, -164.0],
    [-197.0, 199.0],
    [-25.0, 217.0],
    [589.0, -248.0],
    [-269.0, 711.0],
    [185.0, 193.0],
    [315.0, 807.0],
    [-130.0, -43.0],
    [5.0, 3.0],
    [2.0, 17.0],
    [2.0, 5.0],
    [2.0, 3.0],
    [3.0, 1.0],
    [2.0, -1.0],
    [1.0, -1.0],
    [0.0, -1.0],
    [0.0, 0.0],
    [0.0, -2.0],
    [2.0, 2.0],
    [-7.0, 0.0],
    [10.0, -8.0],
    [-3.0, 20.0],
    [6.0, 5.0],
    [14.0, 17.0],
    [-2.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [1.0, 0.0]
])
"""This table contains the periodic terms to compute Pluto's heliocentric
latitude according to Table 37.A in Meeus' book, page 265"""

RADIUS_VECTOR = np.array([
    [66865439.0, 68951812.0],
    [-11827535.0, -332538.0],
    [1593179.0, -1438890.0],
    [-18444.0, 483220.0],
    [-65977.0, -85431.0],
    [31174.0, -6032.0],
    [-5794.0, 22161.0],
    [4601.0, 4032.0],
    [-1729.0, 234.0],
    [-415.0, 702.0],
    [239.0, 723.0],
    [67.0, -67.0],
    [1034.0, -451.0],
    [-129.0, 504.0],
    [480.0, -231.0],
    [2.0, -441.0],
    [-3359.0, 265.0],
    [7856.0, -7832.0],
    [36.0, 45763.0],
    [8663.0, 8547.0],
    [-809.0, -769.0],
    [263.0, -144.0],
    [-126.0, 32.0],
    [-35.0, -16.0],
    [-19.0, -4.0],
    [-15.0, 8.0],
    [-4.0, 12.0],
    [5.0, 6.0],
    [3.0, 1.0],
    [6.0, -2.0],
    [2.0, 2.0],
    [-2.0, -2.0],
    [14.0, 13.0],
    [-63.0, 13.0],
    [136.0, -236.0],
    [273.0, 1065.0],
    [251.0, 149.0],
    [-25.0, -9.0],
    [9.0, -2.0],
    [-8.0, 7.0],
    [2.0, -10.0],
    [19.0, 35.0],
    [10.0, 3.0]
])
"""This table contains the periodic terms to compute Pluto's heliocentric
radius vector according to Table 37.A in Meeus' book, page 265"""

def cal_terms(alpha, mtx, factor):
    return kahan_sum(mtx[:,0]*np.sin(alpha) + mtx[:,1]*np.cos(alpha))*factor


"""
VSOP87 provides methos to calculate Heliocentric coordinates:
    L, the ecliptical longitude (different from the orbtial longitude)
    B, the ecliptical latitude
    R, the radius vector (=distance to the Sun)
"""

def h_rlb_eclip_pluto_j2000(jde):
    """
    Calculates the Heliocentric (ecliptical) polar coordinates of Pluto in J2000 equinox for a Julian Day

    Parameters
    ----------
    jde : float
        The Julian Day 

    Returns
    -------
    np.array 
        A 3 length vector [R, L, B] where
            R : radius vector (distance to the Sun)
            L : the Ecliptical longitude (different from the orbtial longitude)
            B : the Ecliptical latitude
    """
    T = (jde - JD_J2000) / CENTURY
    j = 34.35 + 3034.9097*T
    s = 50.08 + 1222.1138*T
    p = 238.96 + 144.9600*T
    alpha  = np.deg2rad(ARGUMENT[:,0]*j+ARGUMENT[:,1]*s+ARGUMENT[:,2]*p)

    l = 238.958116 + 144.96*T + cal_terms(alpha, LONGITUDE, 1.0e-6)
    b = -3.908239 + cal_terms(alpha, LATITUDE,1.0e-6)
    r = 40.7241346 +  cal_terms(alpha, RADIUS_VECTOR,1.0e-7)

    return np.array([r,deg2rad(l),deg2rad(b)])

def h_xyz_eclip_pluto_j2000(jde):
    """
    Calculates the Heliocentric (ecliptical) cartesian coordinates of Pluto in J2000 equinox for a Julian Day

    Parameters
    ----------
    jde : float
        The Julian Day 

    Returns
    -------
    np.array 
        A 3-vector [x, y, z] where:
            x : 
            y : 
            z : 
    """
    return pipe(jde, h_rlb_eclip_pluto_j2000, cartesianFpolar )


def h_xyz_equat_pluto_j2000(jde):
    """
    Calculates the Equatorial (ecliptical) polar coordinates of Pluto in J2000 equinox for a Julian Day

    Parameters
    ----------
    jde : float
        The Julian Day 

    Returns
    -------
    np.array 
        A 3 length vector [R, L, B] where
            R : radius vector (distance to the Sun)
            L : the Ecliptical longitude (different from the orbtial longitude)
            B : the Ecliptical latitude
    """

    T = (jde - JD_J2000) / CENTURY
    obl = obliquity (T)
    return pipe (h_rlb_eclip_pluto_j2000(jde),
                 cartesianFpolar,   
                 Rx_3d(-obl).dot)


def g_rlb_equat_pluto_j2000(jde):
    """Calculates the Geocentric polar coordinates of Pluto in J2000 equinox for a Julian Day

    Parameters
    ----------
    jde : float
        The Julian Day 

    Returns
    -------
    np.array 
        A 3 length vector [R, L, B] where
            R : radius vector (distance from the Earth)
            L : the Equatorial longitud
            B : the Ecliptical latitude
    """
    g_xyz_equat_pluto = g_xyz_equat_sun_j2000(jde) + h_xyz_equat_pluto_j2000(jde)
    return polarFcartesian(g_xyz_equat_pluto)

if __name__ == "__main__" :
    jde = 2448908.5
    pr_radv(h_rlb_eclip_pluto_j2000(jde))
    #print (h_xyz_equat_pluto_j2000(jde))
    #print(g_xyz_equat_sun_j2000(jde))
    T = (jde - JD_J2000) / CENTURY
    #xyz, _ = planet_heclipt_rv("Pluto",T)
    #rlb = polarFcartesian(xyz)
    #pr_rad (rlb)
    pr_radv(g_rlb_equat_pluto_j2000(jde))

   

  

