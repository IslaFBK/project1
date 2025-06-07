#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:39:40 2020

@author: shni2598
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep
#%%
#fig, [ax1,ax2,ax3,ax4] = plt.subplots(4,1)
fig, ax = plt.subplots(4,1)

aa=np.arange(5); bb=np.arange(5)

ax[0].plot(aa,label='sensory')
ax[1].plot(bb,label='associate')

ax[2].plot(aa,label='sensory')
ax[3].plot(bb,label='associate')
for axi in ax:
    axi.legend()
#%%
plt.legend()
#%%
plt.legend()
#lms = im1 + im2
#labs = [l.get_label() for l in lms]
#ax1.legend(lms, labs)#, loc=0)
#ax2.legend()
#%%
loop_num = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/twoarea/'

data_two = mydata.mydata()
data_two.load(datapath+'data%d.file'%loop_num)
#%%
start_time = 4e3; end_time = 6e3
data_two.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_two.a1.param.Ne, window = 10)
data_two.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_two.a2.param.Ne, window = 10)

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)
#data.a1.ge.get_centre_mass(detect_pattern=True)
#pattern_size2 = data.a1.ge.centre_mass.pattern_size.copy()
#pattern_size2[np.invert(data.a1.ge.centre_mass.pattern)] = 0
title = ''
frames = data_two.a1.ge.spk_rate.spk_rate.shape[2]
# ani = firing_rate_analysis.show_pattern(data_two.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
#                                         show_pattern_size=True, pattern_ctr=data_two.a1.ge.centre_mass.centre_ind, \
#                                             pattern_size=pattern_size2)
stim = [np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[5,5],[5]]]




#%%
ani = show_pattern(data_two.a1.ge.spk_rate.spk_rate, data_two.a2.ge.spk_rate.spk_rate, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, show_pattern_size=False)


#%%
def show_pattern(spkrate1, spkrate2=None, frames = 1000, start_time = 0, interval_movie=10, anititle='', show_pattern_size=False, pattern_ctr=None, pattern_size=None,\
                 stim=None, adpt=None):
    '''
    create amination for the firing pattern of network
    input:
    spkrate1, spkrate2: (N*N*t array)
    frames: number of frames of amination
    start_time: the real-time of the simulation that the first frame of 'spkrate' cooresponds to
    anititle: set the title of amination
    '''
    if spkrate2 is None:
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = plt.figure(figsize = (6,6))
        ax1= fig.add_subplot(111, label="1",frame_on=False)
        ax1.axis('off')
        divider = make_axes_locatable(ax1)
        cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
            
        cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
        camp_adapt = np.array([138/255,43/255,226/255,1.])
        cmap_c = np.array([1.,0.,0.,1.])
        cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        cmap = np.vstack((camp_adapt,cmap_stimulus,cmap_c,cmap_spk(range(7))))
        cmap = mpl.colors.ListedColormap(cmap)
        #cmap.set_under('red')
        bounds = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) - 0.5
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 
        
        cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),
                                        spacing='proportional',
                                        orientation='horizontal') #horizontal vertical
        cb.ax.set_xticklabels(['adpt','stim','ctr', 0, 1, 2, 3, 4, 5, 6])
        cb.set_label('number of spikes')
        
        #titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
        titleaxes = divider.append_axes("top", size="5%", pad=0.01)
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(spkrate1.shape[2]) + start_time
        
        value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)

        if show_pattern_size:
            circle_pattern = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.,color='r',fill=False)
            ax1.add_patch(circle_pattern)
        # def init():
        #     value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
        #     title.set_text(u"time: {} ms".format(time_title[0]))
        #     # if show_pattern_size:
        #     #     circle_pattern = plt.Circle([pattern_ctr[0,1],pattern_ctr[0,0]],pattern_size[0], lw=1.5,color='r',fill=False)
        #     #     ax2.add_patch(circle_pattern)
        #     #     return value1,title,circle_pattern
        #     return value1,title,
        # stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]]
        # #adpt = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6.1,6.1],[6.1]]]]
        # adpt = None
        #ax_init_loc_stim = [None]*2
        if stim is not None:
            loca_n_stim, max_trial_stim, current_state_stim, current_trial_stim, trial_state_stim, all_done_stim, circle_stim = \
                [None], [None], [None], [None], [None], [None], [None] 
            for ax_i in range(1):
                #*ax_init_loc_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_stim[ax_i], max_trial_stim[ax_i], current_state_stim[ax_i], current_trial_stim[ax_i], \
                trial_state_stim[ax_i], all_done_stim[ax_i], circle_stim[ax_i], ax1 = init_loca(ax1, stim[ax_i], 'g')
            circle_all = []
            for cir in circle_stim:
                if cir is not None:
                    circle_all += cir
        if adpt is not None:
            loca_n_adpt, max_trial_adpt, current_state_adpt, current_trial_adpt, trial_state_adpt, all_done_adpt, circle_adpt = \
                [None], [None], [None], [None], [None], [None], [None] 
            for ax_i in range(1):
                #*ax_init_loc_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_adpt[ax_i], max_trial_adpt[ax_i], current_state_adpt[ax_i], current_trial_adpt[ax_i], \
                trial_state_adpt[ax_i], all_done_adpt[ax_i], circle_adpt[ax_i], ax1 = init_loca(ax1, adpt[ax_i], 'tab:purple')
            if 'circle_all' not in locals():
                circle_all = []
            for cir in circle_adpt:
                if cir is not None:
                    circle_all += cir        
        def updatev(i):
            value1.set_array(spkrate1[:,:,i])
            #value2.set_array(spk2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            # if show_pattern_size:
            #     circle_pattern.set_center([pattern_ctr[i,1],pattern_ctr[i,0]])
            #     circle_pattern.set_radius(pattern_size[i])
            #     print('size')
            #     return value1, title,circle_pattern#, value2
            if show_pattern_size:
                circle_pattern.center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                circle_pattern.radius = pattern_size[i]
                if stim is not None:
                    detect_event(i, 1, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                    # for ax_i in range(1):
                    #     if stim[ax_i] is not None:
                    #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                    #         for n in range(loca_n_stim[ax_i]):
                    #             if not all_done_stim[ax_i][n]:
                    #                 if i == stim[ax_i][1][n][current_trial_stim[ax_i][n],trial_state_stim[ax_i][n]]:
                    #                     if current_state_stim[ax_i][n] == False:
                    #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                    #                         circle_stim[ax_i][n].radius = stim[ax_i][2][n][current_trial_stim[ax_i][n]]#pattern_size[i]
                    #                         trial_state_stim[ax_i][n] += 1
                    #                         current_state_stim[ax_i][n] = True
                    #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                    #                     else:
                    #                         circle_stim[ax_i][n].radius = 0
                    #                         trial_state_stim[ax_i][n] -= 1
                    #                         current_trial_stim[ax_i][n] += 1
                    #                         current_state_stim[ax_i][n] = False
                    #                         if max_trial_stim[ax_i][n] == current_trial_stim[ax_i][n]:
                    #                             all_done_stim[ax_i][n] = True
                    #                             print(ax_i,n,'all done')
                    #                             print(ax_i,all_done_stim[ax_i][n])
                    #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                    #                         #sleep(1)
                if adpt is not None:
                    detect_event(i, 1, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                                 trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                
                    # for ax_i in range(1):
                    #     if adpt[ax_i] is not None:
                    #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                    #         for n in range(loca_n_adpt[ax_i]):
                    #             if not all_done_adpt[ax_i][n]:
                    #                 if i == adpt[ax_i][1][n][current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n]]:
                    #                     if current_state_adpt[ax_i][n] == False:
                    #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                    #                         circle_adpt[ax_i][n].radius = adpt[ax_i][2][n][current_trial_adpt[ax_i][n]]#pattern_size[i]
                    #                         trial_state_adpt[ax_i][n] += 1
                    #                         current_state_adpt[ax_i][n] = True
                    #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                    #                     else:
                    #                         circle_adpt[ax_i][n].radius = 0
                    #                         trial_state_adpt[ax_i][n] -= 1
                    #                         current_trial_adpt[ax_i][n] += 1
                    #                         current_state_adpt[ax_i][n] = False
                    #                         if max_trial_adpt[ax_i][n] == current_trial_adpt[ax_i][n]:
                    #                             all_done_adpt[ax_i][n] = True
                    #                             print(ax_i,n,'all done')
                    #                             print(ax_i,all_done_adpt[ax_i][n])
                    #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                if stim is not None or adpt is not None:
                    return (value1, *circle_all, circle_pattern, title) #, circle_adpt[ax_i][1]
                else: return value1, title, circle_pattern,
                
            else:
                if stim is not None:
                    detect_event(i, 1, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                if adpt is not None:                    
                    detect_event(i, 1, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                                 trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                
                if stim is not None or adpt is not None:
                    return (value1, *circle_all, title)
                else:
                    return value1, title,
        
        #value1.set_clim(vmin=0, vmax=6)
        # ax1.axis('off')
        #if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev,  frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
    else:
        fig, ax = plt.subplots(1,2)
        #fig.suptitle('sensory to association strength: %.2f\nassociation to sensory strength: %.2f'
        #             %(scale_e_12[i],scale_e_21[j]))
        #ax1.set_title('sensory')
        #ax2.set_title('association')
        cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
        camp_adapt = np.array([138/255,43/255,226/255,1.])
        cmap_c = np.array([1.,0.,0.,1.])
        cmap_stimulus = np.array([88/255,150/255.,0.,1.])
        cmap = np.vstack((camp_adapt,cmap_stimulus,cmap_c,cmap_spk(range(7))))
        cmap = mpl.colors.ListedColormap(cmap)
        #cmap.set_under('grey')
        bounds = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) - 0.5
        
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 
        
        cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),#+0.5,
                                        spacing='proportional',
                                        orientation='horizontal') #horizontal vertical
        cb.ax.set_xticklabels(['adpt','stim','ctr', 0, 1, 2, 3, 4, 5, 6])
        cb.set_label('number of spikes')
        
        titleaxes = fig.add_axes([0.3, 0.85, 0.4, 0.05])
        titleaxes.axis('off')
        title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
        time_title = np.arange(spkrate1.shape[2]) + start_time
        
        value1=ax[0].matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
        value2=ax[1].matshow(spkrate2[:,:,0], cmap=cb.cmap, norm=cb.norm)
        

        # stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],\
        #         [np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]]
        stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],\
                None]
        adpt = [None,\
                [np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6,6],[6]]]]
        #ax_init_loc_stim = [None]*2
        if stim is not None:
            loca_n_stim, max_trial_stim, current_state_stim, current_trial_stim, trial_state_stim, all_done_stim, circle_stim = \
                [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2 
            for ax_i in range(2):
                #*ax_init_loc_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_stim[ax_i], max_trial_stim[ax_i], current_state_stim[ax_i], current_trial_stim[ax_i], \
                trial_state_stim[ax_i], all_done_stim[ax_i], circle_stim[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
            circle_all = []
            for cir in circle_stim:
                if cir is not None:
                    circle_all += cir
        if adpt is not None:
            loca_n_adpt, max_trial_adpt, current_state_adpt, current_trial_adpt, trial_state_adpt, all_done_adpt, circle_adpt = \
                [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2, [None]*2 
            for ax_i in range(2):
                #*ax_init_loc_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], stim[ax_i], 'g')
                loca_n_adpt[ax_i], max_trial_adpt[ax_i], current_state_adpt[ax_i], current_trial_adpt[ax_i], \
                trial_state_adpt[ax_i], all_done_adpt[ax_i], circle_adpt[ax_i], ax[ax_i] = init_loca(ax[ax_i], adpt[ax_i], 'tab:purple')
            if 'circle_all' not in locals():
                circle_all = []
            for cir in circle_adpt:
                if cir is not None:
                    circle_all += cir
                            
        def updatev(i):
            value1.set_array(spkrate1[:,:,i])
            value2.set_array(spkrate2[:,:,i])
            title.set_text(u"time: {} ms".format(time_title[i]))
            if stim is not None:
                detect_event(i, 2, stim, loca_n_stim, all_done_stim, current_trial_stim, \
                             trial_state_stim, current_state_stim, circle_stim, max_trial_stim)                
                # for ax_i in range(2):
                #     if stim[ax_i] is not None:
                #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                #         for n in range(loca_n_stim[ax_i]):
                #             if not all_done_stim[ax_i][n]:
                #                 if i == stim[ax_i][1][n][current_trial_stim[ax_i][n],trial_state_stim[ax_i][n]]:
                #                     if current_state_stim[ax_i][n] == False:
                #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                #                         circle_stim[ax_i][n].radius = stim[ax_i][2][n][current_trial_stim[ax_i][n]]#pattern_size[i]
                #                         trial_state_stim[ax_i][n] += 1
                #                         current_state_stim[ax_i][n] = True
                #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                #                     else:
                #                         circle_stim[ax_i][n].radius = 0
                #                         trial_state_stim[ax_i][n] -= 1
                #                         current_trial_stim[ax_i][n] += 1
                #                         current_state_stim[ax_i][n] = False
                #                         if max_trial_stim[ax_i][n] == current_trial_stim[ax_i][n]:
                #                             all_done_stim[ax_i][n] = True
                #                             print(ax_i,n,'all done')
                #                             print(ax_i,all_done_stim[ax_i][n])
                #                         print(ax_i,current_trial_stim[ax_i][n],trial_state_stim[ax_i][n],current_state_stim[ax_i][n],all_done_stim[ax_i][n])
                #                         #sleep(1)
            if adpt is not None:
                detect_event(i, 2, adpt, loca_n_adpt, all_done_adpt, current_trial_adpt, \
                             trial_state_adpt, current_state_adpt, circle_adpt, max_trial_adpt)                

                # for ax_i in range(2):
                #     if adpt[ax_i] is not None:
                #         #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
                #         for n in range(loca_n_adpt[ax_i]):
                #             if not all_done_adpt[ax_i][n]:
                #                 if i == adpt[ax_i][1][n][current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n]]:
                #                     if current_state_adpt[ax_i][n] == False:
                #                         #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                #                         circle_adpt[ax_i][n].radius = adpt[ax_i][2][n][current_trial_adpt[ax_i][n]]#pattern_size[i]
                #                         trial_state_adpt[ax_i][n] += 1
                #                         current_state_adpt[ax_i][n] = True
                #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])
                #                     else:
                #                         circle_adpt[ax_i][n].radius = 0
                #                         trial_state_adpt[ax_i][n] -= 1
                #                         current_trial_adpt[ax_i][n] += 1
                #                         current_state_adpt[ax_i][n] = False
                #                         if max_trial_adpt[ax_i][n] == current_trial_adpt[ax_i][n]:
                #                             all_done_adpt[ax_i][n] = True
                #                             print(ax_i,n,'all done')
                #                             print(ax_i,all_done_adpt[ax_i][n])
                #                         print(ax_i,current_trial_adpt[ax_i][n],trial_state_adpt[ax_i][n],current_state_adpt[ax_i][n],all_done_adpt[ax_i][n])

                return (value1, value2, *circle_all, title) #, circle_adpt[ax_i][1]
            else:
                return value1, value2, title,
        
        ax[0].axis('off')
        ax[1].axis('off')

        fig.suptitle(anititle)
        ani=animation.FuncAnimation(fig, updatev, frames=frames, interval=interval_movie, blit=True)    # frames=spk1.shape[2]
        return ani
#%%
stim = [np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]
#%%
def init_loca(ax, stim, color='g'):
        if stim is not None:  
            #print('True')
            stim[0] = np.array(stim[0])
            if len(np.shape(stim[0])) == 1:
                loca_n = 1
                stim[0] = np.array(stim[0]).reshape(-1,2)                
            else: 
                loca_n = np.shape(stim[0])[0]
                stim[0] = np.array(stim[0])
            
            for j in range(len(stim[1])):
                stim[1][j] = np.array(stim[1][j])
                #print(type(stim[1][j]))
                if len(stim[1][j].shape) == 1:
                    stim[1][j] = stim[1][j].reshape(-1,2)
                    print(stim[1][j])
            for j in range(len(stim[2])):
                stim[2][j] = np.array(stim[2][j])  
                       
            max_trial = []
            for trial in stim[1]:
                max_trial.append(np.shape(trial)[0]) 
                print(max_trial)
                
            current_state = np.array([False]*loca_n)
            current_trial = np.array([0]*loca_n)
            all_done = np.array([False]*loca_n)
            trial_state = np.array([0]*loca_n)
            circle = []            
            for n in range(loca_n):
#                circle.append(plt.Circle([stim[0][n,1],stim[0][n,0]],stim[2][n][0], lw=1.,color='r',fill=False))
                circle.append(plt.Circle([stim[0][n,1],stim[0][n,0]],0, lw=1.,color=color,fill=False))               
                ax.add_patch(circle[n])
        else:
            loca_n = None
            max_trial = None
            current_state = None
            current_trial = None
            trial_state = None
            all_done = None
            circle = None
        
        return loca_n, max_trial, current_state, current_trial, trial_state, all_done, circle,  ax

#%%
def detect_event(frame, ax_num, event, loca_n, all_done, current_trial, trial_state, current_state, circle, max_trial):
    for ax_i in range(ax_num):
        if event[ax_i] is not None:
            #loca_n, max_trial, current_state, current_trial, trial_state, all_done, stim_circle = ax_init_loc_stim[ax_i]
            for n in range(loca_n[ax_i]):
                if not all_done[ax_i][n]:
                    if frame == event[ax_i][1][n][current_trial[ax_i][n],trial_state[ax_i][n]]:
                        if current_state[ax_i][n] == False:
                            #stim_circle[n].center = [pattern_ctr[i,1],pattern_ctr[i,0]]
                            circle[ax_i][n].radius = event[ax_i][2][n][current_trial[ax_i][n]]#pattern_size[i]
                            trial_state[ax_i][n] += 1
                            current_state[ax_i][n] = True
                            print(ax_i,current_trial[ax_i][n],trial_state[ax_i][n],current_state[ax_i][n],all_done[ax_i][n])
                        else:
                            circle[ax_i][n].radius = 0
                            trial_state[ax_i][n] -= 1
                            current_trial[ax_i][n] += 1
                            current_state[ax_i][n] = False
                            if max_trial[ax_i][n] == current_trial[ax_i][n]:
                                all_done[ax_i][n] = True
                                print(ax_i,n,'all done')
                                print(ax_i,all_done[ax_i][n])
                            print(ax_i,current_trial[ax_i][n],trial_state[ax_i][n],current_state[ax_i][n],all_done[ax_i][n])



#%%
ani = show_pattern(data_two.a1.ge.spk_rate.spk_rate, data_two.a2.ge.spk_rate.spk_rate, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, show_pattern_size=False)
#%%
ani = show_pattern(data_two.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, \
                                        show_pattern_size=True, pattern_ctr=data_two.a1.ge.centre_mass.centre, pattern_size=pattern_size2,\
                                        stim=[[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]]], \
                                        adpt=[[np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6.1,6.1],[6.1]]]])
#%%
data_two.a1.ge.get_centre_mass(detect_pattern=True)
data_two.a1.ge.centre_mass.centre.shape
pattern_size2 = data_two.a1.ge.centre_mass.pattern_size.copy()
pattern_size2[np.invert(data_two.a1.ge.centre_mass.pattern)] = 0
#%%
fig, ax = plt.subplots(1,1)
cir = plt.Circle([0,0], 0.5)
ax.add_patch(cir)
lis = [0,1]
#%%
cir = change_cir(cir, lis)
#%%
def change_cir(cir, lis):
    cir.radius = 0.2
    lis[0] = 1
    pass
    #return cir
#%%
import firing_rate_analysis
#%%
ani = firing_rate_analysis.show_pattern(data_two.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, \
                                        show_pattern_size=True, pattern_ctr=data_two.a1.ge.centre_mass.centre, pattern_size=pattern_size2,\
                                        stim=[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]], \
                                        adpt=[np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6,6],[6]]])
#%%
ani = firing_rate_analysis.show_pattern(data_two.a1.ge.spk_rate.spk_rate, data_two.a2.ge.spk_rate.spk_rate, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, show_pattern_size=False,\
                                        stim = [[np.array([[31,31],[-0.5,-0.5]]), [np.array([[0,200],[300,400]]), np.array([0,200])], [[6,6],[6]]],None],
                                        adpt = [None,[np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6,6],[6]]]])
    
#%%
adpt=[np.array([[31,31],[-0.5,-0.5]]), [np.array([[100,200],[400,500]]), np.array([100,200])], [[6.1,6.1],[6.1]]]    


