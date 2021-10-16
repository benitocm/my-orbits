"""
This module contains the tests for timeconv function
"""
# Standard library imports


# Third party imports
from pytest import approx
#https://www.scivision.dev/pytest-approx-equal-assert-allclose/
import numpy as np


# Local application imports
from myorbit.planets.vsop87 import *
from myorbit.planets.pluto import *
from myorbit.coord import mk_co_equat2, Coord, EQUAT2_TYPE
from myorbit.data_catalog import CometElms
from myorbit.util.time import datetime2jd
#from myastro.keplerian import g_rlb_equat_body_j2000
#from myastro.ephem import g_rlb_equat_body_j2000 
from myorbit.util  import time as tc


def equals_cood (c1,c2,abs=None,rel=None):
    assert c1.v == approx(c2.v,abs=abs,rel=rel)
    assert c1.equinox == c2.equinox
    assert c1.type == c2.type

def test_hrlb_eclip_eqxdate():
    # pag 219 Meedus
    jde = 2448976.5
    rlb = h_rlb_eclip_eqxdate("Venus", jde, tofk5=False)
    assert rlb == approx(np.array([0.72460168, 0.45577737,-0.04573815]), abs=1e-8)


def test_grlb_equat_body_j2000_1():
    # pag 233 Meedus
    ENCKE = CometElms(name="2P/Encke",
            epoch_name=None ,
            q =  2.2091404*(1-0.8502196) ,
            e = 0.8502196 ,
            i_dg = 11.94524 ,
            Node_dg = 334.75006 ,
            w_dg = 186.23352 ,
            tp_str = "19901028.54502",
            equinox_name = "J2000")

    jde = datetime2jd(1990,10,6,0,0,0)
    # TODO Keplerian needs to be imported
    #c = Coord(g_rlb_equat_body_j2000(jde,ENCKE),"", EQUAT2_TYPE)
    c = None
    equals_cood(c, mk_co_equat2("10h34m14.152s","19°09m31s","",r=0.8242810837210319),abs=1e-4)


def test_grlb_equat_body_j2000_2():
    HALLEY = CometElms(name="1P/Halley",
                epoch_name=None ,
                q =  0.5870992 ,
                e = 0.9672725 ,
                i_dg = 162.23932 ,
                Node_dg = 58.14397 ,
                w_dg = 111.84658 ,
                tp_str = "19860209.43867",
                equinox_name = "B1950")

    jde = tc.datetime2jd(1985,11,15,0,0,0)
    # TODO Keplerian needs to be imported
    #c = Coord(g_rlb_equat_body_j2000(jde,HALLEY),"", EQUAT2_TYPE)
    c = None
    equals_cood(c, mk_co_equat2("4h00m40.226s","22°04m27s","",r=0.7368872829651026),abs=1e-5)


def test_g_xyz_equat_sun_eqxdate():    
    #pag 172 Meeus
    jde = 2448908.5
    c1 = g_xyz_equat_sun_eqxdate(jde,tofk5=True)  
    assert c1 == approx(np.array([-0.93799634, -0.3116537 , -0.13512068]), abs=1e-8)

def test_g_xyz_sun ():
    # pag 175 Meeus
    jde = 2448908.5
    c1 = g_xyz_vsopeclip_sun_j2000(jde)
    assert c1 == approx(np.array([-0.93739691, -0.3413352, -0.00000336]), abs=1e-6)
    c1 = g_xyz_equat_sun_j2000(jde)
    assert c1 == approx(np.array([-0.93739707,-0.31316724, -0.13577841]), abs=1e-6)
    
