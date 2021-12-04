"""This module contains contants values shared between several modules
"""

# Third party imports
import numpy as np


# Product of grativational constant times and the solar mass AU^3/days^2
# This is equivalent to k^2 (Heliocentric Gaussian constant)
# When we are working with sattelites, the units Km/seconds but in Astronomy
# the AU and days are used.
#GM = 2.959122083e-4 
GM = 0.0002959122082322128 
k_2 = GM
k_gauss = 0.01720209895
c_light = 173.14  # [AU/day]

# In the book, Orbital Mechanics, they define mu=G*(M1+m2) where M1 is the mass of th
# the heavy objet and m2 is the object that is orbiting M1. If m2 is an small object
# m2 is negligible and mu=G*M1, that is the same as G*M (for the Sun)

# AU in meters
AU_m = 149597870700
SECONDS_IN_DAY = 3600*24

# Inverse of spped of light in days/AU
INV_C = 0.0057755183

TWOPI = 2*np.pi
PI = np.pi
PI_HALF = np.pi/2

SPAIN_TZ_NAME = "Europe/Madrid"

GM_by_planet = {
    "Mercury" : GM/6023600.0,
    "Venus" : GM/408523.5, 
    "Earth" : GM/328900.5,
    "Mars" : GM/3098710.0,
    "Jupiter" : GM/1047.355,
    "Saturn" : GM /3498.5,
    "Uranus" : GM / 22869.0,
    "Neptune" : GM / 19314.0,
    "Pluto" : GM/3000000.0 
}


CENTENNIAL_PRECESSION_DG = 1.3970
CENTENNIAL_PRECESSION_RAD = np.deg2rad(CENTENNIAL_PRECESSION_DG)

