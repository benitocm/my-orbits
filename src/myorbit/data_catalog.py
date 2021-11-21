"""This module contains functions to read elements file for comets, bodies and planetes
In addition, it provides classes to wrap those data for particular objects

"""
# Standard library imports
from datetime import datetime
from math import isclose
from pathlib import Path
import io
import gzip
import logging

# Third party imports
import pandas as pd
import numpy as np
from numpy import  sqrt,rad2deg, sqrt
from toolz import pipe

# Local application imports
import myorbit.util.timeut as tc
from myorbit.util.timeut import EQX_B1950, EQX_J2000
from myorbit.coord import make_ra, mtx_gauss_vectors, make_lon
from myorbit.util.general import pow
from myorbit.util.constants import TWOPI, GM

from myorbit.init_config import cfg
logger = logging.getLogger(__name__)


def read_planet_elms_for(pl_name, df):
    """Retrieve the orbital elements for a body 

    Parameters
    ----------
    body_name : str
        Name of the body 
    df : pd.Dataframe
        Internal dataframe that contains the  orbital data for a lot of comets read from a JPL file,

    Returns
    -------
    BodyElms
        Orbital elements of the body wrapped in the BodyElms structure. 
        If the body is not found, None is returned.
    """
    row = df[df.name==pl_name]
    if row.empty:
        #logger.error(f'The object {comet_name} does not exist in the Comets database')
        return 
    row = row.to_dict('records')[0]
    return BodyElms(name = row['name'],
                     epoch_name = tc.mjd2epochformat(2451545.0),
                     e = row['e'],
                     i_dg = row['i_dg'],
                     Node_dg = row['Node_dg'],
                     w_dg = row['w_dg'],
                     M_dg = row["M_dg"],
                     a= row["a"]
                     )


class BodyElms:

    """Orbital elements for asteroids
    """
    def __init__(self, name, epoch_name, a, e, i_dg, Node_dg, w_dg, M_dg, tp_mjd=None, equinox_name="J2000"):
        self.name= name
        # Epoch of the elements represented as the Modified Julian Date (MJD), which is defined as the Julian date - 2400000.5
        # In JPL, an example of this datum is 49400 (Halley) that corresponds to the date 1994/02/17. That means that orbital data
        # are mostly updated for that especially the Mean Anomaly, i.e, the Mean Anomaly is the mean anomaly of the body 
        # for that date. That is very important for perturbed methods because they start from this point and go backward or forward
        # That is different from the equinox. The equinox is a time reference for the coordinate that normally are expressed in J2000 equinox
        self.epoch_name = epoch_name
        # This field will have the epoch in MJD, i.e., instead of str, this field contains the integer value
        self.epoch_mjd = pipe(tc.epochformat2jd(self.epoch_name),tc.jd2mjd)
        #Semimajor axis of the orbit (asteroids only).
        self.a = a
        # Eccentricity of the orbit
        self.e = e
        #Inclination of the orbit with respect to the J2000 ecliptic plane.
        self.i = np.deg2rad(i_dg)
        # Longitude of the ascending node (J2000-Ecliptic)
        self.Node = np.deg2rad(Node_dg)
        # Argument of perihelion (J2000-Ecliptic).
        self.w = np.deg2rad(w_dg)
        #Mean anomaly at epoch (asteroids only), i.e., the mean anomaly of the boyd at the time indicated by the epoch_mjd
        self.M0 = np.deg2rad(M_dg)  # The Mean anomaly at epoch_mjd
        # Time of perihelion passage (comets only)
        self.tp_mjd = tp_mjd
        # Equinox used for the coordinates, by default J2000 but in the cases of B1950 are used
        self.eqx_name = equinox_name
        # Century of the equinox
        self.T_eqx0 = tc.T(self.eqx_name)
        # Period of the body in days
        if self.a < 0 :
            # Hyperbolic body so there is no period
            self.period_in_days = None
        else:
            self.period_in_days = TWOPI*sqrt(pow(self.a,3)/GM)
        # Matrix 
        self.mtx_PQR = mtx_gauss_vectors(self.Node,self.i,self.w)

    def calc_M(self, t_mjd) :
        """Computes the Mean Anomaly 

        Parameters
        ----------
        t_mjd : float
            Time of the computation

        Returns
        -------
        float
            Mean anomaly [rads]
        """
        M = (t_mjd - self.epoch_mjd)*TWOPI/self.period_in_days
        M += self.M0
        return tc.reduce_rad(M,to_positive=True)        

    @classmethod
    def in_radians(cls, name, epoch_name, a, e, i_rad, Node_rad, w_rad, M_rad, equinox_name):   
        """[summary]
        Parameters
        ----------
        name : str
            [description]
        epoch_name : str
            [description]
        a : float
            [description]
        e : float, optional
            [description]
        i_rad : float, optional
            [description]
        Node_rad : float
            [description]
        w_rad : float
            [description]
        M_rad : float
            [description]
        equinox_name : str
            [description]

        Returns
        -------
        [type]
            [description]
        """
        
        return cls(name, epoch_name,a,e,np.rad2deg(i_rad),np.rad2deg(Node_rad),np.rad2deg(w_rad),np.rad2deg(M_rad),equinox_name)        

    def as_dict(self):
        d = dict() 
        d['a'] = self.a
        d['e'] = self.e
        d['i_dg'] = rad2deg(self.i)
        d['Node_dg'] = rad2deg(self.Node)
        d['w_dg'] = rad2deg(self.w)
        d['epoch_mjd'] = self.epoch_mjd
        return d

    def __str__(self):

        """Return the string represtantion of a BodyElms instance

        Returns
        -------
        str
            Representation of a BodyElms instance
        """        
        s = []
        s.append(f'Elements for {self.name}')
        s.append(f'            epoch: {self.epoch_name}')
        s.append(f'     equinox name: {self.eqx_name}')
        s.append(f'                a: {self.a} AU')
        s.append(f'                e: {self.e}')
        s.append(f'                i: {np.rad2deg(self.i)} dg')
        s.append(f'             Node: {np.rad2deg(self.Node)} dg')
        s.append(f'                w: {np.rad2deg(self.w)} dg')
        s.append(f'                M: {np.rad2deg(self.M0)} dg')
        s.append(f'        epoch mjd: {self.epoch_mjd} day')
        s.append(f'            T eq0: {self.T_eqx0}')
        s.append(f'    Period (days): {self.period_in_days}')
        #s.append(f'          mtx_PQR: {self.mtx_PQR()}')
        return '\n'.join(s)


class CometElms:

    """[summary]
    """

    def __init__(self, name,  epoch_name, q, e, i_dg, Node_dg, w_dg, tp_str, equinox_name="J2000"):
        """[summary]

        Parameters
        ----------
        name : str
            [description], by default ""
        epoch_name : str
            [description], by default ""
        q : float, optional
            [description]
        e : float, optional
            [description]
        i_dg : float, optional
            [description]
        Node_dg : float, optional
            [description]
        w_dg : float, optional
            [description]
        tp_str : str
            [description], 
        equinox_name : str, optional
            [description], by default "J2000"
        """

        # Name of the comet
        self.name=name
        # Epoch of the elements represented as the Modified Julian Date (MJD), which is defined as the Julian date - 2400000.5
        # In JPL, an example of this datum is 49400 (Halley) that corresponds to the date 1994/02/17. That means that orbital data
        # are mostly updated for that especially the Mean Anomaly, i.e, the Mean Anomaly is the mean anomaly of the body 
        # for that date. That is very important for perturbed methods because they start from this point and go backward or forward
        # That is different from the equinox. The equinox is a time reference for the coordinate that normally are expressed in J2000 equinox
        self.epoch_name = epoch_name
        # This field will have the epoch in MJD, i.e., instead of str, this field contains the integer value
        self.epoch_mjd = pipe(tc.epochformat2jd(self.epoch_name),tc.jd2mjd) if self.epoch_name is not None else None
        #Eccentricity of the orbit.
        self.e = e
        # Perihelion distance (comets only).
        self.q = q
        # Semimajor axis of the orbit (asteroids only) so in the case of comet we tried to calculate it
        self.a = None
        if np.abs(1-e) < 1.e-7:
            self.a = None
        else :
            self.a = q / (1-e)
        # Inclination of the orbit with respect to the J2000 ecliptic plane.
        self.i = np.deg2rad(i_dg)
        # Longitude of the ascending node (J2000-Ecliptic).
        self.Node = np.deg2rad(Node_dg)
        # Argument of perihelion (J2000-Ecliptic).
        self.w = np.deg2rad(w_dg)
        # Time of perihelion passage (comets only) as MJD
        self.tp_mjd = _yyyymmdd_ddd2mjd (tp_str)
        # Time of perihelion passage (comets only) as JD
        self.tp_jd = tc.mjd2jd(self.tp_mjd)
        # Equinox used for the coordinates, by default J2000 but in the cases of B1950 are used
        self.eqx_name = equinox_name
        # Century of the equinox
        self.T_eqx0 = tc.T(self.eqx_name)
        # Matrix
        self.mtx_PQR = mtx_gauss_vectors(self.Node,self.i,self.w)


    def calc_M(self, t_mjd) :
        """Computes the mean anomaly as a function a t, t0 and a, i.e., not depending on the
        period of the orbit=1) and semi-major axis

        Parameters
        ----------
        t_mjd : float
            Time of the computation in Modified Julian Date

        Returns
        -------
        float
            Mean anomaly in radians
        """
        M = sqrt(GM)*(t_mjd-self.tp_mjd)/np.float_power(self.a,1.5)
        return tc.reduce_rad(M,to_positive=True)
    

    def __str__(self):

        """Return the string represtantion of a CometElms instance

        Returns
        -------
        str
            Representation of a CometElms instance
        """
        
        s = []
        s.append(f'Elements for {self.name}')
        s.append(f'     equinox name: {self.eqx_name}')
        s.append(f'            T eq: {self.T_eqx0}')
        s.append(f'        epoch mjd: {self.epoch_mjd} day')
        epoch_date_str = pipe(self.epoch_mjd,tc.mjd2jd,tc.jd2str_date) if self.epoch_mjd is not None else "None"
        s.append(f'       epoch date: {epoch_date_str} ')
        s.append(f'                q: {self.q} AU')
        s.append(f'                a: {self.a} AU')
        s.append(f'                e: {self.e}')
        s.append(f'                i: {np.rad2deg(self.i)} dg')
        s.append(f'             Node: {np.rad2deg(self.Node)} dg')
        s.append(f'                w: {np.rad2deg(self.w)} dg')
        s.append(f'               Tp: {self.tp_mjd} mjd')
        s.append(f'               Tp: {tc.mjd2epochformat(self.tp_mjd)}')
        #s.append(f'          mtx_PQR: {self.mtx_PQR()}')
        return '\n'.join(s)        

def read_planets_orbital_elements (fn) :
    """Read a csv file that contains the orbital elements of the planets

    Parameters
    ----------
    fn : str
        file path of the .csv file

    Returns
    -------
    pd.Dataframe
        Dataframe with the data read from the .csv file.
        The angles are provided in radians.
    """
    df = pd.read_csv(fn,sep='|',header=0, comment='#')
    df['name'] = df['name'].str.strip()
    #cols=['i_dg','w_dg','Node_dg','n_cy','M_dg']
    #df[cols] = df[cols].apply(lambda s : s.map(np.deg2rad))  
    #df = df.rename(columns={'Node_dg': 'Node_rad', 'i_dg': 'i_rad', 'w_dg':'w_rad'})
    #df = df.set_index("name")
    return df
    

def read_ELEMENTS_file(fn):
    """Read a compressed file (downloaded from JPL) that contains the orbital elements for a lot of bodies

    Parameters
    ----------
    fn : str
        file path of the input file

    Returns
    -------
    pd.Dataframe
        Dataframe with the data read from the input file
    """
    headers = []
    ranges = None    
    with gzip.open(fn,'rt') as f:
        logger.info('Reading %s ...', fn)
        for idx, line in enumerate(f):
            if idx == 0:
                headers = filter(lambda tok: len(tok)!=0, line.strip().split(' '))
            elif idx == 1:
                ranges = filter(lambda tok: len(tok)!=0, line.strip().split(' '))
            else :
                break
        f.seek(0, 0)
        remaining_fcontent = io.StringIO(f.read())
    headers = list(headers)[1:] 
    fro = 0
    col_specs =[]
    for tok in ranges:
        to = fro+len(tok)
        col_specs.append((fro,to))
        fro = to+1
    df = pd.read_fwf(remaining_fcontent, names=headers, header=None, dtype = {"Tp":object}, colspecs=col_specs,skiprows=[0,1])
    df['Name'] = df['Name'].str.strip()
    cols=['Epoch','e','i','w','Node']
    if 'a' in df.columns:
        cols.append('a')
    df[cols] = df[cols].apply(lambda s : s.astype(np.float64))
    return df


def _yyyymmdd_ddd2mjd (str_v):
    return pipe(tc.epochformat2jd(str_v[0:4]+'.'+str_v[4:6]+'.'+str_v[6:8]+str_v[8:]), tc.jd2mjd)    

def read_body_elms_for(body_name, df):
    """Retrieve the orbital elements for a body 

    Parameters
    ----------
    body_name : str
        Name of the body 
    df : pd.Dataframe
        Internal dataframe that contains the  orbital data for a lot of comets read from a JPL file,

    Returns
    -------
    BodyElms
        Orbital elements of the body wrapped in the BodyElms structure. 
        If the body is not found, None is returned.
    """
    row = df[df.Name==body_name]
    if row.empty:
        return
    row = row.to_dict('records')[0]
    return BodyElms(name = row['Name'],
                   epoch_name = tc.mjd2epochformat(row['Epoch']),
                   a = row['a'],
                   e = row['e'],
                   i_dg = row['i'],
                   Node_dg = row['Node'],
                   w_dg = row['w'],
                   M_dg = row['M'])


def read_comet_elms_for(comet_name, df) :
    """Retrieve the orbital elements for a comet

    Parameters
    ----------
    comet_name : str
        Name of the comet
    df : pd.Dataframe
        Internal dataframe that contains the  orbital data for a lot of bodies read from a JPL file,

    Returns
    -------
    BodyElms
        Orbital elements of the body wrapped in the BodyElms structure. 
        If the body is not found, None is returned.
    """

    row = df[df.Name==comet_name]
    if row.empty:
        #logger.error(f'The object {comet_name} does not exist in the Comets database')
        return 
    row = row.to_dict('records')[0]
    return CometElms(name = row['Name'],
                     epoch_name = tc.mjd2epochformat(row['Epoch']),
                     q = row['q'],
                     e = row['e'],
                     i_dg = row['i'],
                     Node_dg = row['Node'],
                     w_dg = row['w'],
                     tp_str= row['Tp'])

# These dataframes are read when the module is imported.
DF_COMETS = read_ELEMENTS_file(Path(cfg.get('general','comets_file_path')))
DF_BODIES = read_ELEMENTS_file(Path(cfg.get('general','numbered_asteroids_file_path')))
DF_PLANETS = read_planets_orbital_elements(Path(cfg.get('general','planets_orb_elements_file_path')))

####################################################################
#Some objects are defined for other modules to use.
####################################################################

APOFIS = BodyElms(name="99942 Apophis",
                epoch_name="2008.09.24.0",
                a = .9224383019077086	,
                e = .1911953048308701	,
                i_dg = 3.331369520013644 ,
                Node_dg = 204.4460289189818	,
                w_dg = 126.401879524849	,
                M_dg = 180.429373045644	,
                equinox_name = "J2000")


B_2013_XA22 = BodyElms(name="2013 XA22",
                epoch_name="2020.05.31.0",
                a = 1.100452156382869	,
                e = .2374314858572631		,
                i_dg = 1.960911442205992	 ,
                Node_dg = 82.58938621175157		,
                w_dg = 258.157582490417		,
                M_dg = 295.092371879095		,
                equinox_name = EQX_J2000)

HALLEY_J2000 = read_comet_elms_for("1P/Halley", DF_COMETS)    

HALLEY_B1950 = CometElms(name="1P/Halley",
            epoch_name="1986.02.19.0" ,
            q =  0.5870992 ,
            e = 0.9672725 ,
            i_dg = 162.23932 ,
            Node_dg = 58.14397 ,
            w_dg = 111.84658 ,
            #tp_str = "19860209.43867",
            tp_str = "19860209.44",
            equinox_name = EQX_B1950)

CERES_B1950 = BodyElms(name="Ceres",
                epoch_name="1983.09.23.0",
                a = 2.7657991,
                e = 0.0785650,
                i_dg = 10.60646,
                Node_dg = 80.05225,
                w_dg = 73.07274,
                M_dg = 174.19016,
                equinox_name = EQX_B1950)


CERES_J2000 = read_body_elms_for("Ceres", DF_BODIES)     

# Elliptical comet
C2012_CH17 = read_comet_elms_for("C/2012 CH17 (MOSS)", DF_COMETS)   

# Hyperbolic comet
C_2020_J1_SONEAR = read_comet_elms_for("C/2020 J1 (SONEAR)", DF_COMETS) 

# Parabolic comet:
C_2018_F3_Johnson = read_comet_elms_for("C/2018 F3 (Johnson)", DF_COMETS) 

# Near parabolic comet:
C_2011_W3_Lovejoy = read_comet_elms_for("C/2011 W3 (Lovejoy)", DF_COMETS) 

# Parabolics with problem
C_2007_M5_SOHO = read_comet_elms_for("C/2007 M5 (SOHO)", DF_COMETS) 
C_2003_M3_SOHO = read_comet_elms_for("C/2003 M3 (SOHO)", DF_COMETS) 


def change_date_format(date_str):
    datetime_obj = datetime.strptime(date_str, "%Y-%b-%d")
    return datetime_obj.strftime('%Y/%m/%d')


#DATA = StringIO("""date col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 col13 col14 col15
# 2018-Jun-27 00:00     10 18 38.42 +20 09 06.3   8.84   6.80 3.01457925911114  17.5854844  54.6441 /T  18.8632
# 2018-Jun-29 00:00     10 21 38.70 +19 48 51.4   8.85   6.79 3.03475546300404  17.3474713  53.5420 /T  18.5910
#""")

def read_jpl_data(DATA):
    df = pd.read_csv(DATA, sep="\s+", dtype={"col6":object}, index_col=False) 
    df['date'] = df['date'].map(change_date_format)
    df['ra_hh'] = df['col3'].astype(np.int32)
    df['ra_mm'] = df['col4'].astype(np.int32)
    df['ra_ss'] = df['col5'].astype(np.float32)
    df['col6'] = df['col6'].map(str).str.strip()
    df['sign_de'] = np.where(df['col6'].str[0]=='-',-1,1)
    df['de_dg'] = df['col6'].astype(np.int32).abs()
    df['de_mm'] = df['col7'].astype(np.int32)
    df['de_ss'] = df['col8'].astype(np.float32)
    df['r_AU_1'] = df['col11'].astype(np.float64)
    df['de_sign'] = np.where(df.de_dg>=0,1,-1)
    df['ra_1'] = df.apply(lambda x: tc.hms2dg(x['ra_hh'],x['ra_mm'],x['ra_ss']), axis=1).map(np.deg2rad)
    df['de_1'] = df['sign_de']* df.apply(lambda x: tc.dgms2dg(x['de_dg'],x['de_mm'],x['de_ss'],x['de_sign']), axis=1).map(np.deg2rad)
    cols = ['date','ra_1','de_1','r_AU_1']
    return df[cols].copy()
    

#TESTDATA = StringIO("""col1 col2 col3 col4 col5 col6 col7 col8 
#0  2018/06/27     95.1  168.3  10.6  2.5640  10h18m45.264s  +20°08'30"  3.01395840
#1  2018/06/29     97.0  168.8  10.6  2.5644  10h21m45.488s  +19°48'15"  3.03414295
#""")

def read_my_df(DATA)    :
    df = pd.read_csv(DATA, sep="\s+")
    df['date'] = df['col1']
    df['r_AU_2'] = df['col8'].astype(np.float64)
    df['ra_2'] = df['col6'].map(make_ra)
    df['de_2'] = df['col7'].str.replace("'","m").str.replace('"',"s").map(make_lon)
    cols=['date','ra_2','de_2','r_AU_2']
    return df[cols].copy()

                   
if __name__ == "__main__" :
    import logging.config
    logging.config.fileConfig(CONFIG_INI, disable_existing_loggers=False)    
    print (HALLEY_J2000)
    #print ("For Ceres body\n\n")
    #elm = read_body_elms_for("Ceres",DF_BODIES)
    #print (elm)
    #print ("For Halley Comet\n\n")
    #elm = read_comet_elms_for("1P/Halley",DF_COMETS)
    #print (elm)


    
