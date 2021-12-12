
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
from . import data_catalog as dc
from .util.constants import TWOPI, GM, GM_by_planet


def my_dfdt(t, Y, G, m1, m2):     
    """[summary]

    Parameters
    ----------
    t : [type]
        [description]
    Y : [type]
        [description]
    G : [type]
        [description]
    m1 : [type]
        [description]
    m2 : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """    
    
    x1 = Y[0:3]
    x2 = Y[3:6]
    v1 = Y[6:9]
    v2 = Y[9:12]
    r_3  = np.linalg.norm(x2-x1)**3
    acc1 = G*m2*(x2 - x1)/r_3
    acc2 = G*m1*(x1 - x2)/r_3
    return np.concatenate((v1,v2,acc1,acc2))


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
            lines.append(self.ax.plot([], [], [], '--', c=colors[i], label=name,lw=.8))
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
        for line, pt in zip(self.lines, self.pts):
            line.set_data([], [])
            line.set_3d_properties([])
            pt.set_data([], [])
            pt.set_3d_properties([])
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



if __name__ == "__main__" :
    None
