# import from plugins/matplotlib_view.py
"""Show how to create a custom matplotlib view in the GUI."""

from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas
from numpy import genfromtxt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import colorcet as cc
import numpy as np
import warnings
import time
import sys
import os

warnings.filterwarnings("ignore")

axis_list = []
plot_handles = []
nodata=False

try:
    events = genfromtxt('events.csv', delimiter=',')
except OSError:       
    sys.stderr.write("EventView: events.csv not found in: "+str(os.getcwd())+"\n")
    nodata=True

def _make_default_colormap():
    """Return the default colormap, with custom first colors."""
    colormap = np.array(cc.glasbey_bw_minc_20_minl_30)
    # Reorder first colors.
    colormap[[0, 1, 2, 3, 4, 5]] = colormap[[3, 0, 4, 5, 2, 1]]
    # Replace first two colors.
    colormap[0] = [0.03137, 0.5725, 0.9882]
    colormap[1] = [1.0000, 0.0078, 0.0078]
    return colormap

class EventView(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl  # use matplotlib instead of OpenGL (the default)
    def __init__(self, c=None):
        """features is a function (cluster_id => Bunch(data, ...)) where data is a 3D array."""
        super(EventView, self).__init__()
        self.controller = c
        self.model = c.model
        self.supervisor = c.supervisor
        self.cmap=_make_default_colormap()
    
    def on_request_similar_clusters(self,cid=None):
        self.on_select()

    def on_select(self, cluster_ids=(), **kwargs):
        if nodata:
            return

        global axis_list,plot_handles
        cluster_ids=self.supervisor.selected
        self.cluster_ids=cluster_ids
        self.nclusts = len(cluster_ids)

        if axis_list:
            axis_diff = (self.nclusts-len(axis_list)//2)*2
            if axis_diff<0:
                axis_list = axis_list[0:(len(axis_list))+axis_diff]
                plot_handles = plot_handles[0:len(plot_handles)+axis_diff//2]

        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return

        for i,d in enumerate(np.arange(start=1,stop=self.nclusts*2+1)):
            if d%2 == 0:
                setattr(self,'canvas.ax'+str(d), plt.subplot(2*self.nclusts,1,d,sharex=axis_list[i-1]))
            else:
                setattr(self,'canvas.ax'+str(d), plt.subplot(2*self.nclusts,1,d))

            if (len(axis_list)-1)<i:
                axis_list.append(getattr(self,'canvas.ax'+str(d)))
            else:
                axis_list[i]=(getattr(self,'canvas.ax'+str(d)))
            axis_list[i].cla()

        t1=time.time()
        ttime=0

        for i,d in enumerate(cluster_ids):
            rasters,activity,yrast,ntrials,nevents=self.get_spikes(d)
            t=(time.time())
            axis_list[i*2+1].scatter(rasters,yrast,c=[self.cmap[i]],
                marker='|',s=np.ones(len(rasters))*.4,alpha=0.8)
            axis_list[i*2+1].axvline(x=0,color='white',alpha=.5)
            hist, bins = np.histogram(activity,weights=np.ones(nevents)*(50/ntrials),range=(-5,5),bins=250)
            axis_list[i*2].plot(bins[:-1],hist,color=self.cmap[i])
            axis_list[i*2].axvline(x=0,color='white',alpha=.5)
            axis_list[i*2].set_xticks(np.linspace(-2,3,9))
            axis_list[i*2].set_xlim(left=-2,right=3)
            axis_list[i*2+1].set_ylim(bottom=0,top=ntrials)
            axis_list[i*2].set_ylim(bottom=0,top=None)
            self.fix_axis(axis_list[i*2],10)
            self.fix_axis(axis_list[i*2+1],10)

        print(ttime,'plotting')
        t=time.time()
        # Use this to update the matplotlib figure.
        self.canvas.show()
        print(time.time()-t,'update')
        print(time.time()-t1,'total')
        return


    def fix_axis(self,ax,textsize):
        ax.tick_params(axis='x', labelsize=textsize)
        ax.tick_params(axis='y', labelsize=textsize)
        ax.xaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(False)
    
    def get_spikes(self,clust):
        spikes = self.model.get_cluster_spikes(clust)
        spike_times =np.array(self.model.spike_times[spikes])
        rasters = np.array([])
        yrast = np.array([])
        activity = []
        last_ind = 0
        for i,d in enumerate(events):
            if d<np.amax(spike_times) and d>np.amin(spike_times):
                st = spike_times[last_ind:]
                temp1 = st-(d+5)
                ind1 = np.abs(temp1).argmin()
                if temp1[ind1]>0:
                    ind1-=1
                temp2 = st-(d-5)
                ind2 = np.abs(temp2).argmin()
                if temp2[ind2]<0:
                    ind2+=1
                temp=st[ind2:ind1]-d
                last_ind=ind1
                rasters=np.append(rasters,temp)
                yrast=np.append(yrast,np.ones(len(temp))*i)
                activity.extend(temp)
        return rasters,np.array(activity),yrast,np.amax(yrast),len(activity)


class EventPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_event_view():
            """A function that creates and returns a view."""
            return EventView(c=controller)

        controller.view_creator['EventView'] = create_event_view