"""This module contains functions to read elements file for comets, bodys and planetes
In addition, it provides classes to wrap those data for particular objects

"""
# Standard library imports
from datetime import datetime
from math import isclose
from pathlib import Path
import logging


# Third party imports
import pandas as pd
import numpy as np
from numpy import  sqrt,rad2deg, sqrt
import toolz as tz
from toolz import pipe
import io
import gzip

# Local application imports
import myorbit.util.time as tc
from myorbit.coord import make_ra, mtx_gauss_vectors, make_lon

from myorbit.util.general import pow, GM
from myorbit.util.time import PI, reduce_rad, TWOPI

# Configuration reading
CONFIG_INI=Path(__file__).resolve().parents[2].joinpath('conf','config.ini')
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read(CONFIG_INI)

logger = logging.getLogger(__name__)

class BodyElems:

    def __init__(self, name="", epoch_name="", a=0.0, e=0.0, i_dg=0.0, Node_dg=0.0, w_dg=0.0, M_dg=0.0, tp_mjd=0.0, equinox_name="J2000"):
        self.name= name
        self.epoch_name = epoch_name
        self.epoch_mjd = pipe(tc.epochformat2jd(self.epoch_name),tc.jd2mjd)
        self.a = a
        self.e = e
        self.i = np.deg2rad(i_dg)
        self.Node = np.deg2rad(Node_dg)
        self.w = np.deg2rad(w_dg)
        self.M0 = np.deg2rad(M_dg)  # The Mean anomaly at epoch_mjd
        self.tp_mjd = tp_mjd
        self.eqx_name = equinox_name
        self.T_eqx0 = tc.T(self.eqx_name)
        self.period_in_days = TWOPI*sqrt(pow(self.a,3)/GM)
        self.mtx_PQR = mtx_gauss_vectors(self.Node,self.i,self.w)

    def calc_M(self, t_mjd) -> float :
        """
        """
        M = (t_mjd - self.epoch_mjd)*TWOPI/self.period_in_days
        M += self.M0
        return reduce_rad(M,to_positive=True)        

    @classmethod
    def in_radians(cls, name="", epoch_name="", a=0.0, e=0.0, i_rad=0.0, Node_rad=0.0, w_rad=0.0, M_rad=0.0, equinox_name=""):    
        return cls(name, epoch_name,a,e,np.rad2deg(i_rad),np.rad2deg(Node_rad),np.rad2deg(w_rad),np.rad2deg(M_rad),equinox_name)        

    def as_dict(self):
        d = dict() 
        d['a'] = self.a
        d['e'] = self.e
        d['i_dg'] = rad2deg(self.i)
        d['Node_dg'] = rad2deg(self.Node)
        d['w_dg'] = rad2deg(self.w)
        d['w_dg'] = rad2deg(self.w)
        d['epoch_mjd'] = self.epoch_mjd
        return d

    def __str__(self):
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
    def __init__(self, name="",  epoch_name="", q=0.0, e=0.0, i_dg=0.0, Node_dg=0.0, w_dg=0.0, tp_str="", equinox_name="J2000"):
        self.name=name
        self.epoch_name = epoch_name
        #self.epoch_mjd = pipe(tc.epochformat2jd(self.epoch_name),tc.jd2mjd)
        self.epoch_mjd = pipe(tc.epochformat2jd(self.epoch_name),tc.jd2mjd) if self.epoch_name is not None else None
        self.e = e
        self.q = q
        if isclose(1-e, 0, abs_tol=1e-6):
            self.a = None
        else :
            self.a = q / (1-e)
        self.i = np.deg2rad(i_dg)
        self.Node = np.deg2rad(Node_dg)
        self.w = np.deg2rad(w_dg)
        self.tp_mjd = _yyyymmdd_ddd2mjd (tp_str)
        self.tp_jd = tc.mjd2jd(self.tp_mjd)
        self.eqx_name = equinox_name
        self.T_eqx0 = tc.T(self.eqx_name)
        self.mtx_PQR = mtx_gauss_vectors(self.Node,self.i,self.w)


    def calc_M(self, t_mjd) :
        """Computes the mean anomaly as a function a t, t0 and a, i.e., not depending on the
        period of the orbit=1) and semi-major axis

        Parameters
        ----------
        t_mjd : float
            Time of the compuation in Modified Julian Date

        Returns
        -------
        float
            Mean anomly in radians
        """
        M = sqrt(GM)*(t_mjd-self.tp_mjd)/np.float_power(self.a,1.5)
        return reduce_rad(M,to_positive=True)
    

    def __str__(self):
        """Return the string represtantion of a CometElemts instance

        Returns
        -------
        str
            Representation of a CometElemts instance
        """

        s = []
        s.append(f'Elements for {self.name}')
        s.append(f'     equinox name: {self.eqx_name}')
        s.append(f'            T eq: {self.T_eqx0}')
        s.append(f'        epoch mjd: {self.epoch_mjd} day')
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
    logger.info('Reading %s ...', fn)
    df = pd.read_csv(fn,sep='|',header=0)
    df['name'] = df['name'].str.strip()
    cols=['i','w','Node','n','M']
    df[cols] = df[cols].apply(lambda s : s.map(np.deg2rad))    
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
    #df['Tp_jd']= df['Tp'].map(str).map(lambda v: v[0:4]+'.'+v[4:6]+'.'+v[6:8]+v[8:]).map(co.epochformat2jd)
    #df['epoch_name']= df['Epoch'].map(co.mjd2epochformat)
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
    BodyElems
        Orbital elements of the body wrapped in the BodyElems structure. 
        If the body is not found, None is returned.
    """
    row = df[df.Name==body_name]
    if row.empty:
        logger.error('The body %s does not exist is not found ', body_name)
        return
    row = row.to_dict('records')[0]
    return BodyElems(name = row['Name'],
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
    BodyElems
        Orbital elements of the body wrapped in the BodyElems structure. 
        If the body is not found, None is returned.
    """

    row = df[df.Name==comet_name]
    if row.empty:
        logger.error('The comet %s does not exist is not found ', comet_name)
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
DF_BODYS = read_ELEMENTS_file(Path(cfg.get('general','numbered_asteroids_file_path')))
DF_PLANETS = read_planets_orbital_elements(Path(cfg.get('general','planets_orb_elements_file_path')))

#Some objects

APOFIS = BodyElems(name="99942 Apophis",
                epoch_name="2008.09.24.0",
                a = .9224383019077086	,
                e = .1911953048308701	,
                i_dg = 3.331369520013644 ,
                Node_dg = 204.4460289189818	,
                w_dg = 126.401879524849	,
                M_dg = 180.429373045644	,
                equinox_name = "J2000")


B_2013_XA22 = BodyElems(name="2013 XA22",
                epoch_name="2020.05.31.0",
                a = 1.100452156382869	,
                e = .2374314858572631		,
                i_dg = 1.960911442205992	 ,
                Node_dg = 82.58938621175157		,
                w_dg = 258.157582490417		,
                M_dg = 295.092371879095		,
                equinox_name = "J2000")


def change_date_format(date_str):
    datetime_obj = datetime.strptime(date_str, "%Y-%b-%d")
    return datetime_obj.strftime('%Y/%m/%d')


#DATA = StringIO("""date col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 col13 col14 col15
# 2018-Jun-27 00:00     10 18 38.42 +20 09 06.3   8.84   6.80 3.01457925911114  17.5854844  54.6441 /T  18.8632
# 2018-Jun-29 00:00     10 21 38.70 +19 48 51.4   8.85   6.79 3.03475546300404  17.3474713  53.5420 /T  18.5910
#""")

def read_jpl_data(DATA):
    df = pd.read_csv(DATA, sep="\s+", dtype={"col6":object}) 
    df['date'] = df['date'].map(change_date_format)
    df['ra_hh'] = df['col3'].astype(np.int32)
    df['ra_mm'] = df['col4'].astype(np.int32)
    df['ra_ss'] = df['col5'].astype(np.float32)
    df['col6'] = df['col6'].map(str).str.strip()
    df['sign_de'] = np.where(df['col6'].str[0]=='-',-1,1)
    df['de_dg'] = df['col6'].astype(np.int32).abs()
    df['de_dg'] = df['de_dg']
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
    print ("For Ceres body\n\n")
    elm = read_body_elms_for("Ceres",DF_BODYS)
    print (elm)
    print ("For Halley Comet\n\n")
    elm = read_comet_elms_for("1P/Halley",DF_COMETS)
    print (elm)


    
