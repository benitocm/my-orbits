"""
This module contains the tests for timeconv function
"""
# Standard library imports
# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np
from numpy.random import default_rng

# Local application imports
from myorbit.util.timeut import *
from myorbit.coord import *

#from myastro.orbit import *
#from myastro.keplerian import next_E

def test_polar_cartesian():
    xyzs = [np.array([0,0,0]),
            np.array([1,1,1]),
            np.array([1,1,-1]),
            np.array([1,-1,1]),
            np.array([1,-1,-1]),
            np.array([-1,1,1]),
            np.array([-1,1,-1]),
            np.array([-1,-1,1]),
            np.array([-1,-1,-1])]
    for xyz in xyzs:        
        polar = polarFcartesian(xyz)
        assert xyz == approx(cartesianFpolar(polar))

def test_make_lon ():
    assert make_lon("0°0m0s") == 0
    assert make_lon("2°0m0sW") < 0
    assert make_lon("-2°0m0s") < 0

def equals_cood (c1,c2,abs=None,rel=None):
    assert c1.v == approx(c2.v,abs=abs,rel=rel)
    #assert c1.equinox == c2.equinox
    assert c1.type == c2.type

def test_horiz2equat_1():
    lat_obs = make_lat("38°0m0sN")
    # We remove 
    #horiz = mk_co_horiz("115°0m0s","40°0m0s")
    horiz = mk_co_horiz("295°0m0s","40°0m0s")

    equat = equat1Fhoriz(horiz,lat_obs)
    equals_cood(equat, mk_co_equat1("21h1m53.6s","8°5m2.5s",''),abs=1e-5)

"""
def test_horiz2equat_2():
    rng = default_rng()
    rnd_ab = lambda n,a,b : (b-a)*rng.random((n,))+a
    n = 2
    lats = rnd_ab(n,-90,90)
    alts = rnd_ab(n,-90,90)
    azs = rnd_ab(n,0,360)

    tups = map(horiz2equat,lats,alts,azs)
    Hs, decs = map(list,zip(*tups))
    tups = map(equat2horiz,lats,Hs,decs)
    alts2, azs2 =  map(list,zip(*tups))
    assert alts == approx(alts2)
    assert azs == approx(azs2)

    rng = default_rng()
    rnd_ab = lambda n,a,b : (b-a)*rng.random((n,))+a
    n = 40
    lats = rnd_ab(n,-90,90)
    alts = rnd_ab(n,-90,90)
    azs = rnd_ab(n,0,360)
    tups = map(horiz2equat,lats,alts,azs)
    Hs, decs = map(list,zip(*tups))
    tups = map(equat2horiz,lats,Hs,decs)
    alts2, azs2 =  map(list,zip(*tups))
    assert alts == approx(alts2)
    assert azs == approx(azs2)
    """


"""
def test_eclip2equat():
    lon, lat = tc.dgms2dg(184,36,00),tc.dgms2dg(1,12,00)
    assert eclip2equat(lon,lat) == approx((12.313193170034987, -0.726530584324963))

    rng = default_rng()
    rnd_ab = lambda n,a,b : (b-a)*rng.random((n,))+a
    n = 40
    lons = rnd_ab(n,0,360)
    lats = rnd_ab(n,-90,90)
    tups = map(eclip2equat,lons,lats)
    ras, decs = map(list,zip(*tups))
    tups = map(equat2eclip,ras,decs)
    lons2, lats2 = map(list,zip(*tups))
    assert lons == approx(lons2)
    assert lats == approx(lats2)


def test_equinox_correction():
    assert equinox_correction(12.816667,27.4,1950.0,2000.0) == approx((12.857323608897255, 27.12802708736664))


# Excercies pag 106
def test_ch04Ex01():
    gst = tc.ut2gst(1976,6,5,14,0,0)
    h,m,s, _ = tc.gst2lst(*tc.h2hms(gst),-64)
    ra = tc.ha_lst2ra(tc.hms2h(15,30,15),h,m,s)
    assert tc.h2hms(ra) == approx((11.0, 10.0, 13.548841))

def test_ch04Ex02():
    gst = tc.ut2gst(2015,1,5,12,0,0)
    h,m,s, _ = tc.gst2lst(*tc.h2hms(gst),40)
    ha = tc.ra_lst2ha(tc.hms2h(12,32,6),h,m,s)
    assert tc.h2hms(ha) == approx((9.0, 6.0, 57.6327))   
"""

def test_ch04Ex03():
    lat_obs = make_lat("35°36m5.115sN")
    #Instead of 200° we put 200-180=20° because the book use Az=0 in
    #the north
    horiz = mk_co_horiz("20°10m20s","10°0m0s")
    equat = equat1Fhoriz(horiz,lat_obs)
    equals_cood(equat, mk_co_equat1("1h46m14.676s","-40°34m53.36s",''),abs=1e-6)

def test_ch04Ex04():
     #def equat2horiz(lat_dg:float, H_h:float , dec_dg:float) -> Tuple : 
    lat_obs = make_lat("80°0m0sS")
    equat = mk_co_equat1("7h0m0s","49°54m20s")
    horiz = horizFequat1(equat,lat_obs)
    equals_cood(horiz, mk_co_horiz("267°7m3.71s","-51°28m20.52s"))

def test_ch04Ex05():
    eclip = mk_co_eclip("120°30m30s","00°0m0s")
    T = tc.T("J2000")
    equat = equat2Feclip(eclip,T)
    equals_cood(equat, mk_co_equat2("8h10m50s","20°02m31s","J2000"),abs=1e-4)
    
def test_ch04Ex06():
    equat = mk_co_equat2("11h10m13s","30°05m40s")
    T = tc.T("J2000")
    eclip = eclipFequat(equat,T)
    equals_cood(eclip, mk_co_eclip("156°19m09s","22°41m54s","J2000"), abs=1e-4)

"""
def test_ch04Ex11():
    ra,dec = equinox_correction(tc.hms2h(12,32,6), tc.dgms2dg(30,5,40),1950.0,2000.0)
    assert tc.h2hms(ra) == approx((12,34,34.305790))
    assert tc.dg2dgms(dec) == approx((29,49,7.896504,1))

def test_ch04Ex12():
    ra,dec = equinox_correction(tc.hms2h(12,34,34), tc.dgms2dg(29,49,34),2000.0,2015.0)
    assert tc.h2hms(ra) == approx((12,35,18.391146))
    #assert tc.dg2dgms(dec) == approx((29,44,11,1))

"""

"""
def test_ch04Ex13():
    e, m_anomaly  = 0.00035 , np.deg2rad(5.498078)
    # Solving with Newton
    e_funcs = [next_E] 
    for func in e_funcs:        
        e_anomaly = solve_ke_newton(e, func, m_anomaly, m_anomaly)        
        assert np.rad2deg(e_anomaly) == approx(5.5, abs=1e-02)

    # Solving just Iteraring
    for func in e_funcs:        
        e_anomaly = solve_ke (e, func, m_anomaly)        
        assert np.rad2deg(e_anomaly) == approx(5.5, abs=1e-02)
    
def test_ch04Ex14():
    e, m_anomaly  = 0.6813025 , np.deg2rad(5.498078)
    # Solving with Newton
    e_funcs = [next_E] 
    for func in e_funcs:        
        e_anomaly = solve_ke_newton(e, func, m_anomaly, m_anomaly)        
        assert np.rad2deg(e_anomaly) == approx(16.744355,abs=1e-4)

    # Solving just Iteraring
    #for func in e_funcs:        
    #    e_anomaly = solve_ke(e, func, m_anomaly)        
    #    assert np.rad2deg(e_anomaly) == approx(16.744355,abs=1e-3)

def test_ch04Ex15():
    e, m_anomaly  = 0.85 , np.deg2rad(5.498078)  
    # Solving with Newton
    e_funcs = [next_E] 
    for func in e_funcs:        
        e_anomaly = solve_ke_newton(e, func, m_anomaly, m_anomaly)        
        assert np.rad2deg(e_anomaly) == approx(29.422286,abs=1e-4)
        
    # Solving just Iteraring
    #for f in e_funcs:        
    #    e_anomaly = solve_ke(e, func, m_anomaly)        
    #    assert np.rad2deg(e_anomaly) == approx(29.422286,abs=1e-4)

def test_change_equinox_equat():
    # Meeus book pag 137
    f_epoch="J2000.0"
    t_epoch="2028.11.13.19"
    equat = mk_co_equat2("2h44m12.975s","49°13m39.90s")
    changed_equat = change_equinox_equat(f_epoch,t_epoch,equat)
    equals_cood(changed_equat, mk_co_equat2("02h46m11.331s","+49°20m54.54s",t_epoch), abs=1e-4)
    

def test_change_equinox_eclip():
    f_epoch="J2000.0"
    t_epoch="-214.30.0"
    eclip = mk_co_eclip("149°28m54.984s","1°45m55.76s")
    changed_eclip = change_equinox_eclip(f_epoch,t_epoch,eclip)
    equals_cood(changed_eclip, mk_co_eclip("+118°43m51.01s","+01°36m55.74s",t_epoch), abs=1e-4)
"""
