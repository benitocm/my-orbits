"""
This module contains functions related to time conversions and utilities
"""
# Standard library imports
from typing import Any,Dict,List,Tuple,Sequence
#https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html

from datetime import datetime
from functools import partial
from math import isclose

# Third party imports
import numpy as np
from toolz import pipe
from pytz import timezone
from numpy.polynomial.polynomial import Polynomial
from datetime import datetime
from toolz import compose
from numpy import deg2rad, rad2deg

# Local application imports
from myorbit.util.constants import *

#
# Python's built-in float type has double precision
#

def my_fix(x):
    """Given a float number, return an integer that is less
    than or equal to the argument

    Parameters
    ----------
    x : float
        The value 

    Returns
    -------
    int
        The integer part of the x
    """
    return int(np.modf(x)[1])

def my_frac(x) :
    """Given a float number, return the fraction part as float ignoring
    the sign

    Parameters
    ----------
    x : float
        The value

    Returns
    -------
    float
        The fraction part of x
    """
    return np.abs(np.modf(x)[0])


def reduce_rad(rad, to_positive=False):
    """Reduce an angle to a value between (-TWOPI, TWOPI), i.e. an open interval

    Parameters
    ----------
    rad : float
        angle to reduce [radians]
    to_positive : bool, optional
        Indicates if a negative angle should be converted to its positive value (adding TWOPI), by default False

    Returns
    -------
    float
        The angle reduced [radians]
    """
    remainder = my_frac(rad/TWOPI)*TWOPI
    if isclose(remainder, 0.0, abs_tol=1e-9):
        return 0.0
    if rad >= 0.0 :
        return remainder
    else :
        return -remainder + TWOPI if to_positive else -remainder        

def norm_dg(degrees):
    """Given a angle in degress (positve or negative), computes
    the equivalent angle between [0,360) degrees. In case,
    the angle is negative, the positive version is returned.

    Parameters
    ----------
    degrees : float
        Angle that may include the fractional part [degrees]

    Returns
    -------
    float
        The equivalent angle between [0,360)
    """
    if 0 <= degrees < 360 :
        return degrees
    frac = my_frac(degrees)
    fix =  my_fix(degrees)
    new_alpha = (np.abs(fix) % 360) + frac    
    if isclose(new_alpha, 0, rel_tol=0, abs_tol=1e-9):
        return 0.0
    elif degrees < 0 :
        return 360 - new_alpha        
    else :
        return new_alpha

def norm_rad(alpha):
    cycles = my_fix(alpha/TWOPI)
    new_alpha = alpha - (cycles*TWOPI)
    if isclose(new_alpha, 0.0, rel_tol=0, abs_tol=1e-12) or isclose(new_alpha, TWOPI, rel_tol=0, abs_tol=1e-12):
        return 0.0
    if alpha < 0 :
        new_alpha = TWOPI + new_alpha
    return new_alpha

# angles [-360, 360]
# time   [0,24]


def dgms2dg(dg ,arm = 0, ars =0, sign = 1) : 
    """Convert from degress, minutes and seconds to decimal degrees

    Parameters
    ----------
    dg : float
        The degress value
    arm : float, optional
        The minutes value, by default 0
    ars : float, optional
        The second value, by default 0
    sign : int, optional
        The sign, when +1 means positive, when -1 means negative, by default +1

    Returns
    -------
    float
        The decimal degress
    """
    value = np.abs(dg) + arm/60 + ars/3600
    value = value if sign>0 else -value 
    return value

def dg2dgms(ddg) : 
    """Converts from decimal degress to degrees, minutes and seconds

    Parameters
    ----------
    ddg : float
        A value in decimal degress

    Returns
    -------
    Tuple
        A tuple (dg, min, sec, sign) 
    """

    dminutes, dg =  np.modf(np.abs(ddg))
    dseconds, minutes = np.modf(dminutes*60)
    seconds = np.around(dseconds * 60)
    sign = 1 if ddg > 0 else -1
    if seconds == 60 :
        return (dg,minutes+1,0,sign)
    else :
        return (dg,minutes,dseconds * 60,sign)

def hms2h(h = 0, m = 0, s=0)  : 
    """Converts from hours, minutes and seconds to decimal hours
    Used for Right Ascension within a range  [0h0m0s -> 23h59h59s]

    Parameters
    ----------
    h : float, optional
        hours, by default 0
    m : float, optional
        minute, by default 0
    s : float, optional
        seconds, by default 0

    Returns
    -------
    float
        decimal hours
        None for negative values.
    """
    if 0 <= h < 24 :
        if 0 <= m < 60 :
            if 0 <= s < 60 :
                value = h + m/60 +  s/3600
                return value
    
    

def h2hms(dh) : 
    """Converts from decimal hours to hours, minutes and seconds
    Used for Right Ascension within a range  [0h0m0s -> 23h59h59s]

    Parameters
    ----------
    dh : float
        Decimal hours

    Returns
    -------
    Tuple
        A tuple (h, min, sec, sign) 
        None for negative values
    """
    if dh < 0:
        return None
    dm, h = np.modf(np.abs(dh))
    ds, m = np.modf(dm*60)
    s = np.around(ds * 60)
    if s == 60 :
        return h,m+1,0
    else :
        return h,m,ds*60

def hms2dg(h = 0, m = 0, s =0) : 
    """Converts from  hours, minutes and seconds to decimal degrees   
    Used for Right Ascension within a range  [0h0m0s -> 23h59h59s]

    Parameters
    ----------
    h : float, optional
        hours, by default 0
    m : float, optional
        minutes, by default 0
    s : float, optional
        seconds, by default 0

    Returns
    -------
    float
        [description]
    """
    if 0 <= h < 24 :
        if 0 <= m < 60 :
            if 0 <= s < 60 :
                return hms2h(h,m,s)*15


def dg2h(d)  :
    """Converts from decimal degrees to decimal hours. Used to convert to RA

    # TODO negative values?
    Parameters
    ----------
    d : float
        decimal degrees value. It should be positive

    Returns
    -------
    float
        Decimal hours value (Right Ascension)
    """
    
    return d/15
    
def dms2h (dg, arm = 0, ars =0, sign = 1) -> float : 
    """Convert from degrees, minutes and seconds to decimal hours.
    Used for R.A

    Parameters
    ----------
    dg : float
        Degrees
    arm : float, optional
        Minutes, by default 0
    ars : float, optional
        Seconds, by default 0
    sign : int, optional
        The sign, by default 1

    Returns
    -------
    float
        Decimal hours
    """
    return dg2h(dgms2dg(dg,arm,ars,sign))


def hms2fd(h,m,s):
    """Converts a point in time in a day into the fractional day

    Parameters
    ----------
    h : [type]
        [description]
    m : [type]
        [description]
    s : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return hms2h(h,m,s)/24


def fd2hms(fraction_day):
    """    
    Given a fractional day, returns the the time (h,m,s)
    """ 
    return h2hms(fraction_day*24)

def sec2rads(secs:float) -> float :
    """
    From seconds to radians
    """
    return pipe(dgms2dg(0,0,secs),np.deg2rad)    

# Trigonometric functions where the angle is expresed in degrees
sin_dgms = compose(np.sin,np.deg2rad,dgms2dg)
cos_dgms = compose(np.cos,np.deg2rad,dgms2dg)
tan_dgms = compose(np.tan,np.deg2rad,dgms2dg)

# Trigonometric functions where the angle is expresed decimal degrees
sin_dg = compose(np.sin,np.deg2rad)
cos_dg = compose(np.cos,np.deg2rad)
tan_dg = compose(np.tan,np.deg2rad)


# Trigonometric functions where the angle is expressed as time
sin_hms = compose(np.sin,np.deg2rad,hms2dg)
cos_hms = compose(np.cos,np.deg2rad,hms2dg)
tan_hms = compose(np.tan,np.deg2rad,hms2dg)


def is_gregorian (year, month, day):
    """ Given a date as year, month number and day number, return if
    that date is gregorian or not

    Parameters
    ----------
    year : int
        A year number
    month : [type]
        A month number 1<=month<=12
    day : [type]
        day number 1<=day<=31

    Returns
    -------
    bool
        True if the date is Gregorian. False, otherwise
    """
    if year > 1582:
        return True
    elif year < 1582 :
        return False
    elif month > 10 :
        return True
    elif month < 10 :
        return False
    else :
        return (day>=15)


def datetime2jd(year,month,day,hour=0,minute=0,second=0) :
    """Given a date-time as year number, month number, day number, hour,
    minute and secondd, calculate the corresponding Julian Day.

    A Julian day begins at 12h UT (noon) and not at 0h UT (midnight)
    so whenever the fractional part of a Julian day is 0.0, the time of day
    was noon (UT) for the calendar converted whereas whenever the fractional
    part is 0.5, the time of day was midnight (UT)


    Parameters
    ----------
    year : int
        A year number
    month : int
        A month number 1<=month<=12
    day : int
        A day number 1<=day<=31
    hour : int, optional
        A hour number 0<=hour<=23, by default 0
    minute : int, optional
        A minute number 0<=minute<=59, by default 0, by default 0
    second : int, optional
        A second number 0<=second<=59, by default 0, by default 0

    Returns
    -------
    float
        The julian day
    """
    m = month if (month>2) else month+12
    y = year if (month>2) else year-1
    t = 0.75 if (year<0) else 0
    is_greg = is_gregorian(year,month,day)
    a = my_fix(y/100) if is_greg else 0
    b = 2-a+my_fix(a/4) if is_greg else 0         
    jd = b + my_fix(365.25*y-t)+my_fix(30.6001*(m+1))+day+1720994.5+hms2fd(hour,minute,second)
    return jd

def jd2datetime(julian_day) :
    """Converts a Julian day to the corresponding date-time value, i.e, 
    year, month, day, fractional_day

    Parameters
    ----------
    julian_day : float
        A julian day value

    Returns
    -------
    Tuple
        (year, month, day, fractional_day)
    """
    
    y, mo,fd = jd2datefd(julian_day)
    d = my_fix(fd)    
    return (y,mo,d,*fd2hms(my_frac(fd)))    

def mjd2str_date(mjd):
    """Given a julian day, returns its corresponding date-time as 
    string

    Parameters
    ----------
    mjd : float
        a Modified Julian Day

    Returns
    -------
    str
        The date-time as a string corresponding to a Modified Julian Day.
    """
    return pipe(mjd, mjd2jd, jd2str_date)

def jd2str_date(jd):
    """Given a julian day, returns its corresponding date-time as 
    string

    Parameters
    ----------
    jd : float
        a Julian day

    Returns
    -------
    str
        The date-time as a string corresponding to the julian day.
    """
    tup = jd2datetime(jd)
    return f'{tup[0]:04.0f}/{tup[1]:02.0f}/{tup[2]:02.0f}'


def datefd2jd(year, month, day) :
    """ Converts a date with year, month and day with fractional part into
    the corresponding Julian day

    Parameters
    ----------
    year : int
        A year number
    month : int
        A month number 1<=month<=12
    day : float
        a day value with fractional part

    Returns
    -------
    float
        The julian day
    """
    frac_day = my_frac(day)  
    d = my_fix(day)  
    return datetime2jd(year,month,d, *fd2hms(frac_day))    

def datefd2mjd(year, month, day:float) -> float :
    """ Converts a date with year, month and day with fractional part into
    the corresponding Modified Julian day.  Its bases is  November 17, 1858 at 0h

    Parameters
    ----------
    year : int
        A year number
    month : int
        A month number 1<=month<=12
    day : float
        a day value with fractional part

    Returns
    -------
    float
        The Modified julian day
    """
    return datefd2jd(year,month, day) - 2400000.5


def ymdfh2mjd(year, month, day, frac_day) -> float :
    """
    Computes the modified Julian day (MJD). Its bases is November 17, 1858 at 0h
    """
    return datetime2jd(year,month,day, *fd2hms(frac_day)) - 2400000.5

def mjd2jd(mjd)  : 
    """Converts a Modified Julian Day into a Julian Day

    Parameters
    ----------
    mjd : float
        A Modified Julian Day

    Returns
    -------
    float
        The Julian Day
    """
    return mjd + 2400000.5

def jd2mjd(jd) : 
    """Converts a Julian Day into a Modified Julian Day

    Parameters
    ----------
    mjd : float
        A Julian Day

    Returns
    -------
    float
        The Modified Julian Day
    """
    return jd - 2400000.5

def jd2datefd_v2( julian_day : float ) -> Tuple:
    """
    Given a Julian day calculate its date and time in UT 
    """
    jd = julian_day + 0.5
    i =  my_fix(jd)
    f =  my_frac(jd)    
    if i > 2299160:
        a =  my_fix((i-1867216.25)/36524.25)
        b = i + 1 + a -  my_fix(a/4)
    else :
        a = None
        b = 1
    c = b + 1524
    d = my_fix((c - 122.1)/365.25)
    e = my_fix(365.25*d)
    g = my_fix((c-e)/30.6001)
    day =  c - e + f - my_fix(30.6001*g)
    month = (g - 1) if g < 13.5 else (g - 13)
    year = (d - 4716) if month > 2.5 else (d-4715)
    return year,month,day


    
def jd2datefd( julian_day : float ) -> Tuple:
    """
    Given a Julian day calculate its date and time in UT 
    """
    jd = julian_day + 0.5
    z =  my_fix(jd)
    f =  my_frac(jd)    
    if z < 2299161 :
        a = z
    else :
        alpha = my_fix((z-1867216.25)/36524.25)
        a = z + 1 + alpha - my_fix( alpha/4)
    
    b = a + 1524
    c = my_fix((b- 122.1)/365.25)
    d = my_fix(365.25*c)
    e = my_fix((b-d)/30.6001)
    day = b - d - my_fix(30.6001*e) + f
    if e< 14:
        month = e- 1
    else :
        month = e-13
    if month > 2:
        year = c - 4716
    else :
        year = c - 4715

    return year,month,day

def is_leap_year(year : int ) -> bool :
    return (year%400 == 0) or (( year%4 == 0 ) and ( year%100 != 0))

def datet2elapsed_days(year,month,day):
    t = 1 if is_leap_year(year) else 2
    return  my_fix(275*month/9) - (t * my_fix((month+9)/12)) + day - 30 

def elapsed_days2date(year,ndays):
    a = 1523 if is_leap_year (year) else 1889
    b = my_fix((ndays+a-122.1)/362.25)
    c = ndays + a - my_fix(365.25*b)
    e = my_fix(c/30.6001)
    month = e-1 if e < 13.5 else e-13
    day = c - my_fix(30.6001*e)
    return year,month,day

DAYS = {
    1 : "Monday",
    2 : "Tuesday",
    3 : "Wednesday",
    4 : "Thursday",
    5 : "Friday",
    6 : "Saturday",
    7 : "Sunday"
}

def dayOfweek(year,month,day) :
    jd = datefd2jd(year,month,day)
    a = (jd + 1.5)/7
    b = 7*my_frac(a)
    return DAYS[np.round(b)]


def tz2tz(h,m,s,is_dst=False,from_tz=None,to_tz=None):
    """
    """
    dt = datetime(2020,7,24,h,m,s) if is_dst else datetime(2020,1,24,h,m,s)
    loc_dt = timezone(from_tz).localize(dt)
    dt_ut = loc_dt.astimezone(timezone(to_tz))
    return dt_ut.hour, dt_ut.minute, dt_ut.second
    

lct2ut = partial(tz2tz,to_tz='UTC')
"""
"""

ut2lct = partial(tz2tz,from_tz='UTC')
"""
"""

def norm_hours(h):
    """
    """

    if (h < 0) :
        return h+24,1
    elif (h > 24) :
        return h-24,1
    else :
        return h,0

flatten = lambda tup,value : sum([tup,(value,)],())

def ut_lct_by_lon(h,m,s,lon,adjust_f=None,is_dst=0,sign=None):
    """
    longitude east 0 to 180
    longitude west 0 to -180

    #https://stackoverflow.com/questions/3204245/how-do-i-convert-a-tuple-of-tuples-to-a-one-dimensional-list-using-list-comprehe

    Note that calling norm_hours can change the date of reference, 
    +1 day or -1 day
    """
    hd = hms2h(h,m,s)    
    if is_dst :
        hd = hd + sign
    adjust = sign*adjust_f(lon)
    hd = hd + adjust
    hd, incr = norm_hours(hd)
    return flatten(h2hms(hd),incr)
    

lon_lct2ut = partial(ut_lct_by_lon,adjust_f = lambda lon : np.round(lon/15) ,sign=-1)
"""
"""

ut2lon_lct = partial(ut_lct_by_lon,adjust_f = lambda lon : np.round(lon/15) ,sign=1)
"""
"""

def ut2gst_v2(year,month,day,h,m,s):
    """
    """    
    jd = datefd2jd(year,month,day)
    jd0 = datefd2jd(year,1,0.0)
    n_days = jd - jd0
    t = (jd0 - 2415020.0)/36525.0

    r = 6.6460656 + (2400.051262*t) + (0.00002581 * t * t)
    b = 24 - r + 24 *(year - 1900)
    t0 = (0.0657098*n_days) - b
    ut = hms2h(h,m,s)
    gst = t0 + 1.002738*ut
    gst = norm_hours(gst)
    return h2hms(gst)

def polynomial(a0,a1,a2,a3,a4,x):
    """
    Up to x4
    """
    return a0 + x*(a1+x*(a2+x*(a3+x*a4)))

def ut2gst(year:float, month:float ,day: float,
           h:float, m:float, s:float ) -> float:
    """    
    Calculate GST (Greenwhich Sidereal Time) given
    the UT (Universal Time), date and time
    This is the version of of Meeus's book (pg 88)
    
    Args:
        year: ut year 
        month: ut month [1,12]
        day: ut day [1,31]
        h : ut hours [0,24)
        m : ut minutes [0,60)
        s : ut seconds [0,60)

    Returns :
        The GST in time units [0,24)
    """    
    jd = datetime2jd(year,month,day,h,m,s)
    t = (jd - 2451545)/CENTURY
    a0 = 280.46061837 + 360.98564736629 * (jd-2451545)
    theta0 = polynomial(a0,0,0.000387933,-1/38710000,0,t)
    # theta0 is degrees so it is translated to time units
    return pipe(theta0, norm_dg,dg2h)
    

def gst2ut(year,month,day,h,m,s):
    jd = datefd2jd(year,month,day)
    jd0 = datefd2jd(year,1,0)
    n_days = jd - jd0
    t = (jd0 - 2415020.0)/CENTURY
    p = Polynomial([6.6460656,2400.051262,0.00002581])    
    r = p(t)
    b = 24 - r + 24 *(year - 1900)
    t0 = (0.0657098*n_days) - b
    # The increment is not taken into account
    t0, incr = norm_hours(t0)
    gst = hms2h(h,m,s)
    a = gst - t0
    a = a + 24 if (a < 0) else a
    ut = 0.997270*a
    return flatten(h2hms(ut),incr)


gst2lst = partial(ut_lct_by_lon,adjust_f = lambda lon: lon/15,is_dst=0,sign=1)
"""    
Calculate GST (Greenwhich Sidereal Time) given
LST (Local Sideral Time) and the longitud of the observer (local)

Args:
    h : lst hours [0,24)
    m : lst minutes [0,60)
    s : lst seconds [0,60)
    lon : longitude of the observer in degrees

Returns :
    The LST (float) in time units.
""" 

lst2gst = partial(ut_lct_by_lon,adjust_f = lambda lon: lon/15,is_dst=0,sign=-1)
"""
"""

def lst_now(lon : float) -> Tuple :
    """
    Provides the current local sidereal time based on longitud
    """
    gst_now =  ut2gst(*ut_now())
    return gst2lst(*gst_now,lon)[0:-1]

def ut_now() :
    return datetime.utcnow().timetuple()[0:6]


def format_time(hhmmss,as_int=True):
    if (as_int):
        return '{:02.0f}:{:02.0f}:{:02.0f}'.format(*hhmmss)
    else :
        return '{:02}:{:02}:{:02}'.format(*hhmmss)
    
def ra_lst2ha(ra, h,m,s):
    """
    Calculate the hour angle given the right ascension and LST time
    (Local Sideral Time).
    
    Args:
        ra : Right ascensionHour angle in time units [0,24).
        h : lst time hours [0,24)
        m : lst time minutes
        s : lst time seconds

    Returns :
        The hour angle in time units [0,24)
    """

    lst = hms2h(h,m,s)
    ha = lst - ra
    ha = ha+24 if ha < 0 else ha
    return ha

def ha_lst2ra(ha : float, h:float , m: float, s:float) -> float:
    """
    Calculate the right ascension given the hour angle and LST time
    (Local Sideral Time).
    
    Args:
        ha : Hour angle in time units [0,24).
        h : lst time hours [0,24)
        m : lst time minutes
        s : lst time seconds

    Returns :
        The right ascension in time units [0,24)
    """

    lst = hms2h(h,m,s)
    ra = lst - ha
    ra = ra+24 if ra < 0 else ra
    return ra

def mjd2epochformat(mjd):
    y,mo, fd= jd2datefd(mjd+2400000.5)
    return f'{y}.{mo}.{fd}'

def quadrant (alpha) :
    dg = np.rad2deg(alpha)
    if dg < 0:
        dg = norm_dg(dg)
    if 0 <= dg <= 90 :
        return 1
    elif 90 < dg <= 180 :
        return 2
    elif 180 < dg <= 270 :
        return 3
    else :
        return 4 

def hemisphere (alpha) :
    dg = np.rad2deg(alpha)
    if dg < 0:
        dg = norm_dg(dg)
    if 0 <= dg <= 180 :
        return 1
    else :
        return 2




def epochformat2jd (epoch_name:str) -> float :
    """ 
    Computes the number of Julian centuries from a epoch format data

    Args:
        epoch_name: The name of the epoch following a format:
            "2000.01.04.03" --> year=2000, month=01 fractional_day=04.03

        Normal epoch names :
            1700.0
            1900.0 

        Equinox names
            Julian Epoch J2000 (2000 January 1.5 = JD 2451545.0)
            B1950 (Besselian year) = JD 2433282.423

    Returns :
        A float with nunmber of Julian centuries
    """
   
    epoch = None
    if (epoch_name == "J2000") or (epoch_name == "J2000.0")  :
	    return JD_J2000
    elif epoch_name == "B1950" or (epoch_name == "B1950.0")  :
        return JD_B1950
    else :
        toks = epoch_name.split(".")
        if len (toks) == 2:
            fd = 0
        elif len (toks) ==3:
            fd = float (toks[2])
        elif len (toks) ==4:
            fd = float (toks[2]+'.'+toks[3])
        epoch = (float (toks[0]),float (toks[1]),fd)
    return  datefd2jd(*epoch)

JD_J2000 = datefd2jd(*(2000.0,1.0,1.5))
#JD_B1950 =  2433282.42345905 #datefd2jd(*(1950.0,1.0,0.9235))
JD_B1950 =  2433282.4235 #datefd2jd(*(1950.0,1.0,0.9235))
MDJ_J2000 = jd2mjd(JD_J2000)
CENTURY = 36525.0
T_J2000 = 0.0
EQX_J2000="J2000"
EQX_B1950="B1950"

def T_given_mjd(mjd):
    """ 
    Computes the number of Julian centuries between the modified Julian day given and the 
    and J2000 (2000 January 1 12h, i.e. See pag 32 in Astronmy with your Personal Computer

    J2000 = tc.datefd2jd(2000,1,1.5)

    Args:
        mjd: The Julian day given

        A float with nunmber of Julian centuries
    """
    return (mjd - MDJ_J2000)/CENTURY

    

def T (epoch_name:str, from_epoch_name="J2000") -> float :
    """ 
    Computes the number of Julian centuries between the epoch given (epoch_name) and J2000 (2000 January
    1 12h, i.e. See pag 32 in Astronmy with your Personal Computer

    J2000 = tc.datefd2jd(2000,1,1.5)

    Args:
        epoch_name: The name of the epoch following a format:
            "2000.01.04.03" --> year=2000, month=01 fractional_day=04.03

        Normal epoch names :
            1700.0
            1900.0 

        Equinox names
            Julian Epoch J2000 (2000 January 1.5 = JD 2451545.0)
            B1950 (Besselian year) = JD 2433282.423

    Returns :
        A float with nunmber of Julian centuries
    """
    jd = epochformat2jd(epoch_name)
    return (jd - JD_J2000)/CENTURY




if __name__ == "__main__":
    print (norm_rad(TWOPI))
    #print (datefd2jd(1957,10,4.81))
    #print (ymdfh2mjd(1805, 9, 5, 24.165))
    #print (datetime2jd(1805,9, 5, hour=24.165) )
    #print (dms2d(0,30,30.00,"-"))
    #print ( hms2h (9,14,55.8))
    #print (elapsed_days2date(2005,68))
    #print (dayOfweek(2020,4,22))
    #print (ut2gst(2014,12,12,1,0,0))
    #print (ut2gst(2010,2,7,23,30,00))
    #print (ut2gst(1987,4,10,19,21,00))
    #print (gst2ut(1987,4,10,8,34.0,57.02))
    #print (ut2gst_v2(1987,4,10,19,21,0))
    #print (ut2gst(2004,3,4,4,30,0))
    #print (ut2gst(2004,3,4,4,30,0))
    #print (gst2lst(15,19,6.92,139.80))
    #print (lon_lct2ut(20,0,0,-77,is_dst=False))
    #print (gst2lst(6,26,34,-77))
    #print (gst2lst(2,3,41,-40))
    #print (ut2gst(2001,10,3,6,30,0))
    #print (lct2lst())
    #print (lst_now(-3.8796))
    #print (lst_now(179))
    #print (tz.pipe(179,lst_now,format_time))
    #print (format_time((1,12,3)))
    #print (ut_now())
    #print (ra_lst2ha(3,24,6,18,0,0))
    #print (ha_lst2ra(1,15,0,21,0,0))
    #h1 = hms2h(1,1,1)
    #print (h1)
    #hms1 = h2hms(h1)
    #print (hms1)


    #def h2hms(dh : float) -> Tuple: 
 

    None