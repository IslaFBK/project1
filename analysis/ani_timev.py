#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:49:11 2020

@author: shni2598
"""
import numpy as np
#%%
def show_timev(v1, v2=None, vrange=[[-80,-50]], frames = 1000, start_time = 0, interval_movie=10, anititle=''):#, show_pattern_size=False, pattern_ctr=None, pattern_size=None):
    '''
    create amination for the firing pattern of network
    input:
    spkrate1, spkrate2: (N*N*t array)
    frames: number of frames of amination
    start_time: the real-time of the simulation that the first frame of 'spkrate' cooresponds to
    anititle: set the title of amination
    '''
    if v2 is None:
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize = (6,6))
        ax1= fig.add_subplot(111, label="1",frame_on=False)
        ax1.axis('off')
        divider = make_axes_locatable(ax1)
        cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
        # cb = plt.colorbar(cbaxes)
        #fig.colorbar(img1, cax=cax1)
        # if show_pattern_size:
        #     ax2=fig.add_subplot(111, label="2",frame_on=False)
        #     ax2.set_xlim([-0.5,spkrate1.shape[1]+0.5])
        #     ax2.set_ylim([-0.5,spkrate1.shape[0]+0.5])       
        #     ax2.axis('off')
            
        # cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
        # cmap_c = np.array([1.,0.,0.,1.])
        # cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        # cmap = np.vstack((cmap_stimulus,cmap_c,cmap_spk(range(7))))
        # cmap = mpl.colors.ListedColormap(cmap)
        # #cmap.set_under('red')
        # bounds = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # #cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 
        
        # cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
        #                                 norm=norm,
        #                                 boundaries=bounds,
        #                                 ticks=np.array([-2,-1, 0, 1, 2, 3, 4, 5, 6])+0.5,
        #                                 spacing='proportional',
        #                                 orientation='horizontal') #horizontal vertical
        # cb.ax.set_xticklabels(['stimulus','ctr', 0, 1, 2, 3, 4, 5, 6])
        # cb.set_label('number of spikes')
        
        #titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
        titleaxes = divider.append_axes("top", size="5%", pad=0.01)
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(v1.shape[2]) + start_time
        
        value1=ax1.matshow(v1[:,:,0], vmin = vrange[0][0], vmax = vrange[0][1])#,cmap=cb.cmap, norm=cb.norm)
        cb = plt.colorbar(value1, cax=cbaxes, orientation='horizontal')
        # if show_pattern_size: 
        #     ax2 = fig.add_axes([0, 0, 1, 1]) 
        #     #ax2= plt.subplot(111)
        #     circle = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.5,color='r',fill=False)
        #     ax2.add_patch(circle)
        # if show_pattern_size:
        #     circle = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.,color='r',fill=False)
        #     ax1.add_patch(circle)

        
        def updatev(i):
            value1.set_array(v1[:,:,i])
            #value2.set_array(spk2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            # if show_pattern_size:
            #     circle.set_center([pattern_ctr[i,1],pattern_ctr[i,0]])
            #     circle.set_radius(pattern_size[i])
            #     print('size')
            #     return value1, title,circle#, value2
            # if show_pattern_size:
            #     circle.center = [pattern_ctr[i,1],pattern_ctr[i,0]]
            #     circle.radius = pattern_size[i]
            #print('size')
                # return value1, title,circle,#, value2
            #return title,circle,
            #else:
            return value1, title,
        
        #value1.set_clim(vmin=0, vmax=6)
        # ax1.axis('off')
        #if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev,  frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
    else:
        fig, [ax1,ax2]= plt.subplots(1,2)

        cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03])

        titleaxes = fig.add_axes([0.3, 0.85, 0.4, 0.05])
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(v1.shape[2]) + start_time
        
        value1=ax1.matshow(v1[:,:,0], vmin = vrange[0][0], vmax = vrange[0][1])
        value2=ax2.matshow(v2[:,:,0], vmin = vrange[1][0], vmax = vrange[1][1])
        
        cb = plt.colorbar(value1, cax=cbaxes, orientation='horizontal')
        
        
        def updatev(i):
            value1.set_array(v1[:,:,i])
            value2.set_array(v2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            return value1, value2, title,
        
        ax1.axis('off')
        ax2.axis('off')
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev, frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani

#%%
timeseries1 = np.random.uniform(-80,-50,[5,5,200])
timeseries2 = np.random.uniform(-60,-50,[5,5,200])

#%%
#ani = show_timev(timeseries1, v2=None, vrange=[[-80,-50]], frames = 200, start_time = 0, interval_movie=10, anititle='')#, show_pattern_size=False, pattern_ctr=None, pattern_size=None):
ani = show_timev(timeseries1, v2=timeseries2, vrange=[[-80,-50],[-80,-50]], frames = 200, start_time = 0, interval_movie=10, anititle='')#, show_pattern_size=False, pattern_ctr=None, pattern_size=None):
#%%
analy_type = 'toa1'
datapath = ''
sys_argv = 0#int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)

#%%
start_time = int(5e3); end_time = int(6e3)
#data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
frames = int(end_time - start_time)

ani = firing_rate_analysis.show_timev(data.a1.ge.v[:,start_time:end_time].reshape(63,63,-1),\
                                      data.a1.gi.v[:,start_time:end_time].reshape(32,32,-1), vrange=[[-80,-50],[-80,-50]],\
                                      frames = frames, start_time = start_time, \
                                      interval_movie=20, anititle=title)

