#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:51:11 2020

@author: shni2598
"""
file_num = 13*5 + 1 #int(sys.argv[1])
#file_num=14
scale_e_12 = np.concatenate([np.array([0]),np.arange(0.2,1.4,0.1)])
scale_e_21 = np.concatenate([np.array([0]),np.arange(0.2,1.4,0.1)])
loop_num = -1
for i in range(len(scale_e_12)):
    for j in range(len(scale_e_21)):
        loop_num += 1
        if loop_num == file_num:
            print(i,j)
            break
    else:continue
    break

timepath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli/timev%d%s.mat'#%(area,file_num)
indexpath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli/indexv%d%s.mat'#%(area,file_num)

spkv1 = psa.importmatdata(timepath, indexpath, file_num=file_num, area=1)
spkv2 = psa.importmatdata(timepath, indexpath, file_num=file_num, area=2)

spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkv1, starttime=500*ms, endtime=6000*ms, \
      binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)
spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkv2, starttime=500*ms, endtime=6000*ms, \
      binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)

spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)
spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)

spkrate1[31,31,500:] = -2
spkrate1[62,0,500:] = -2
ani=psa.show_pattern(spkrate1, spkrate2, area_num = 2, frames=4000, stimu_onset=1000, bottom_up=scale_e_12[i], top_down=scale_e_21[j], start_time = 500)
ani=psa.show_pattern(spkrate1, area_num = 1, frames=4000, stimu_onset=1000, bottom_up=scale_e_12[i], top_down=scale_e_21[j], start_time = 500)

#%%
start_time = 500
fig, [ax1,ax2]= plt.subplots(1,2)
#fig.suptitle('sensory to association strength: %.2f\nassociation to sensory strength: %.2f'
#             %(scale_e_12[i],scale_e_21[j]))
#ax1.set_title('sensory')
#ax2.set_title('association')

#        cmap=plt.cm.get_cmap('Blues', 7) # viridis Blues
#        cmap.set_under('red')
#        bounds = [0, 1, 2, 3, 4, 5, 6]

cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
cmap_c = np.array([1.,0.,0.,1.])
cmap_stimulus = np.array([0.,1.,0.,1.])
cmap = np.vstack((cmap_stimulus,cmap_c,cmap_spk(range(7))))
cmap = mpl.colors.ListedColormap(cmap)
#cmap.set_under('red')
bounds = [-2, -1, 0, 1, 2, 3, 4, 5, 6]

norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cbaxes = fig.add_axes([0.2, 0.15, 0.6, 0.03]) 

cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                norm=norm,
                                boundaries=bounds,
                                ticks=bounds,
                                spacing='proportional',
                                orientation='horizontal') #horizontal vertical
cb.set_label('number of spikes')

titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
titleaxes.axis('off')
title = titleaxes.text(0.5,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
time_title = np.arange(spkrate1.shape[2]) + start_time

value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
plt.show()












