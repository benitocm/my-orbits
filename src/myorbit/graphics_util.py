
"""
This module contains functions related to orbit plotting
"""
# Standard library imports

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
from toolz import concat, first
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# Local application imports
import myorbit.data_catalog as dc
from myorbit.two_body import calc_eph_planet
from myorbit.pert_cowels import calc_eph_by_cowells
from myorbit.pert_cowels import calc_eph_by_cowells
from myorbit.util.constants import TWOPI, GM, GM_by_planet

# Local application imports

#https://stackoverflow.com/questions/33587540/simple-matplotlib-animate-not-working
class OrbitsPlot:

    def __init__(self, orbs_data, date_refs, axes_limits, center_label='SUN',center_color='yellow') :
        self.fig = plt.figure()
        self.orbs = orbs_data
        self.date_refs = date_refs
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection='3d')

        n_trajectories = len(self.orbs)

        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, n_trajectories))

        # lines and points initializaiton
        lines = []
        pts = []
        for i, (name, mtx) in enumerate(self.orbs.items()):
            lines.append(self.ax.plot([], [], [], '--', c=colors[i], label=name,lw=.7))
            pts.append(self.ax.plot([], [], [], 'o', c=colors[i]))
        self.lines = list(concat(lines))
        self.pts = list(concat(pts))

        # prepare the axes limits
        self.ax.set_xlim(axes_limits)
        self.ax.set_ylim(axes_limits)
        self.ax.set_zlim(axes_limits)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # The center is plotted
        self.ax.scatter3D(0,0,0, color=center_color, marker='o', lw=8, label=center_label)
            
       
        #   set the legend
        self.ax.legend(loc='upper right', prop={'size': 9})
        #ax.set_title("Tim-Sitze, Orbits of the Inner Planets")
        #animation.writer = animation.writers['ffmpeg']

        axtext = self.fig.add_axes([0.0,0.95,0.1,0.05])
        # turn the axis labels/spines/ticks off
        axtext.axis("off")

        self.time_obj = axtext.text(0.5,0.5, date_refs[0], ha="left", va="top")

    # initialization function: plot the background of each frame
    def init(self):
        print ("Init called !!!!")
        for line, pt in zip(self.lines, self.pts):
            line.set_data([], [])
            line.set_3d_properties([])
            pt.set_data([], [])
            pt.set_3d_properties([])
        print ("Returning form init")
        return self.lines + self.pts

    def animate(self, i):
        for line, pt, mtx in zip(self.lines, self.pts, self.orbs.values()):
            xs = mtx[0:i,0]        
            ys = mtx[0:i,1]
            zs = mtx[0:i,2]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)
            
            x = xs[-1:]
            y = ys[-1:]
            z = zs[-1:]        
            pt.set_data(x, y)
            pt.set_3d_properties(z)
            
            self.time_obj.set_text(self.date_refs[i])

        #ax.view_init(30, 0.3 * i)
        self.fig.canvas.draw()
        return self.lines + self.pts
    
    def start(self,interval=1000, blit=False, repeat=False):
        self.anim = animation.FuncAnimation(self.fig, self.animate, frames=len(self.date_refs), interval=interval, blit=blit, repeat=repeat)
        
    
    
class MyAni(object):
    def __init__(self, size=4800, peak=1.6):
        self.size = size
        self.peak = peak
        self.fig = plt.figure()
        self.x = np.arange(self.size)
        self.y = np.zeros(self.size)
        self.y[0] = self.peak
        self.line, = self.fig.add_subplot(111).plot(self.x, self.y)

    def animate(self, i):
        self.y[i - 1] = 0
        self.y[i] = self.peak
        self.line.set_data(self.x, self.y)
        return self.line,

    def start(self):
        self.anim = animation.FuncAnimation(self.fig, self.animate,
            frames=self.size, interval=20, blit=False)
        

def calc_tp(M_at_epoch, a, epoch_mjd):
    deltaT = TWOPI*a*np.sqrt(a/GM)*(1-M_at_epoch/TWOPI)
    return deltaT + epoch_mjd        

PLANET_NAMES= [x.lower() for x in GM_by_planet.keys()]
def calc_interval(obj_names):
    tps=[]
    for name in obj_names:
        if not isinstance(name, str):
            obj = name
            tp_mjd = obj.tp_mjd
            if obj.tp_mjd is None:
                tp_mjd = calc_tp(obj.M0,obj.a,obj.epoch_mjd)
            tps.append(tp_mjd)
            continue
        if name.lower() in PLANET_NAMES:
            print (f'{name} is a planet')
            continue
        obj = dc.read_comet_elms_for(name,dc.DF_COMETS)        
        if obj is not None:
            tps.append(obj.tp_mjd)
        else :
            obj = dc.read_body_elms_for(name,dc.DF_BODIES)
            if obj is not None:
                tp_mjd = calc_tp(obj.M0,obj.a,obj.epoch_mjd)
                tps.append(tp_mjd)
            else:
                print (f'Object {name} not found')
    if len(tps) > 0:
        return min(tps), max(tps)



def calc_orbits_heliocentric_data(eph, obj_names):
    """
    Computes the orbits of the planets, minor bodys and comets 
    
    Args:
        eph : EphemerisData
        planets : List of name of planets
        minor_bodys : List of names of minor bodys or orbital elements itself
        comets : List of names of comets bodys or orbital elements itself

    Returns :
        orbs : A dictionary where the key is the name of the body and value is a
               matrix of n,3 (n rows per 3 cols) with the heliocentric coordinates h_x, h_y, h_z
               and the index is the date of corresponding to the position.
        date_refs :  list of the dates where the heliocentric coordinates were calculated
        
    """    
    # orbs is a dictionary where the key is the name of the object (planet, asteroids or comet)
    # and the value is the dataframe with the ephemeris data.
    orbs = {}
    dfs = []
    for name in obj_names:
        if not isinstance(name, str):
            # Assumed that this is a BodyElms or CometElms 
            obj = name
            df  = calc_eph_by_cowells(obj,eph, include_osc=False)
            orbs[obj.name] = df
            dfs.append(df) 
            continue           
        if name.lower() in PLANET_NAMES:
            df = calc_eph_planet(name, eph)
            orbs[name] = df
            dfs.append(df)
        else :
            obj = dc.read_comet_elms_for(name,dc.DF_COMETS)        
            if obj is not None:
                df  = calc_eph_by_cowells(obj,eph, include_osc=False)
                orbs[name] = df
                dfs.append(df)
            else :
                obj = dc.read_body_elms_for(name,dc.DF_BODIES)
                if obj is not None:
                    df  = calc_eph_by_cowells(obj,eph, include_osc=False)
                    orbs[name] = df
                    dfs.append(df)
                else :
                    print (f"Object {name} not found")
    # Assumed that the ['date'] colum of each ephemeris are the same for every object so
    # we get the list of dates from the first object.
    first_key= list(orbs.keys())[0]
    date_refs = orbs[first_key]['date'].to_list()
    cols=['h_x','h_y','h_z']    
    for k, df in orbs.items():
        # For each object, the ecliptic (heliocentric) coordinates are kept and
        # transformed to a matrix with shape (len(date_refs), 3)
        #    [[x1,y1,z1],
        #     [x2,y2,z2],
        #      ....
        #     [xn,yn,zn]]
        # for each key in the obr object, the value will be a nx3 matrix with the heliocentric coordinates
        orbs[k] = df[cols].to_numpy()     
    return orbs, dfs, date_refs

def change_reference_frame(heliocentric_orbs, name):
    orbs_from_obj = dict()
    # A new orbs object is created changing the frame of reference to the Earth
    for body_name in filter(lambda x : x.lower()!=name.lower(), heliocentric_orbs.keys()):
        orbs_from_obj[body_name] = heliocentric_orbs[body_name] - heliocentric_orbs[name]    
    return orbs_from_obj


if __name__ == "__main__" :
    from myorbit.ephemeris_input import EphemrisInput
    OBJS=['Ceres','Pallas','Juno','Vesta',dc.APOFIS]
    tp_min, tp_max = calc_interval(OBJS)
    eph = EphemrisInput.from_mjds(tp_min-150, tp_max+150, "5 00.0", "J2000" )
    print(eph)
    orbs, dfs, date_refs = calc_orbits_datav3(eph, OBJS)
    
    eph = EphemrisInput(from_date="2009.01.01.0",
                        to_date = "2010.12.01.0",
                        step_dd_hh_hhh = "5 00.0",
                        equinox_name = "J2000")

    #PLANETS = ['Earth','Mercury','Venus','Mars']
    #PLANETS = ['Jupiter','Saturn','Uranus','Neptune', 'Pluto']
    PLANETS = ['Earth','Mars']
    #PLANETS = []
    #PLANETS = ['Jupiter','Saturn']
    #MINOR_BODYS = []
    #MINOR_BODYS = ['Ceres','Pallas','Juno','Vesta']
    #MINOR_BODYS = ['Ceres',APOFIS]
    #MINOR_BODYS = ['Ceres']
    MINOR_BODYS = []
    #MINOR_BODYS=['2002 NN4','2010 NY65', dc.B_2013_XA22]
    #COMETS = ['1P/Halley','2P/Encke','10P/Tempel 2','C/1995 O1 (Hale-Bopp)']
    #COMETS = ['C/2019 Q4 (Borisov)']
    #COMETS = ['D/1993 F2-A (Shoemaker-Levy 9)']
    #COMETS = ['C/1988 L1 (Shoemaker-Holt-Rodriquez)'] #, 'C/1980 E1 (Bowell)','C/2019 Q4 (Borisov)']
    #COMETS = ['C/2019 Q4 (Borisov)']
    COMETS = []

    orbs,  date_refs = calc_orbits_data(eph, PLANETS, MINOR_BODYS, COMETS)
