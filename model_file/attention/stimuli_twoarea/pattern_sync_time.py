# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:02:37 2020

@author: nishe
"""

import brian2.numpy_ as np
from brian2.only import *
import matplotlib.pyplot as plt
import post_analysis as psa
from analysis.ignore_shortevent import ignore_shortevent
import connection as cn
#%%
file_num = 13*12 + 1 #int(sys.argv[1])
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

starttime = 0*ms; endtime = 6000*ms
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkv1, starttime, endtime, binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)
spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkv2, starttime, endtime, binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)

#%%
spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)
spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)

spkrate1[31,31,1000:] = -2
spkrate1[62,0,1000:] = -2
ani=psa.show_pattern(spkrate1, spkrate2, area_num = 2, frames=4000, stimu_onset=1000, bottom_up=scale_e_12[i], top_down=scale_e_21[j], start_time = 0)
#ani=psa.show_pattern(spkrate1, area_num = 1, frames=4000, stimu_onset=1000, bottom_up=scale_e_12[i], top_down=scale_e_21[j], start_time = 500)

#%%
#[[-31.5, -31.5],[0, 0]])
sti1 = [0, 0]; sti2 = [-31.5, -31.5]

centre_ind1[:,0] -= 31
centre_ind1[:,0] *= -1
centre_ind1[:,1] -= 31

centre_ind2[:,0] -= 31
centre_ind2[:,0] *= -1
centre_ind2[:,1] -= 31

spatial_threshold = 10
ctr1tosti1 = cn.coordination.lattice_dist(centre_ind1, 62, sti1)
ctr1tosti2 = cn.coordination.lattice_dist(centre_ind1, 62, sti2)
ctr1closesti1 = ctr1tosti1 <= spatial_threshold
ctr1closesti2 = ctr1tosti2 <= spatial_threshold

ctr2tosti1 = cn.coordination.lattice_dist(centre_ind1, 62, sti1)
ctr2tosti2 = cn.coordination.lattice_dist(centre_ind1, 62, sti2)
ctr2closesti1 = ctr2tosti1 <= spatial_threshold
ctr2closesti2 = ctr2tosti2 <= spatial_threshold

ctrdist = cn.coordination.dist_periodic_pointwise(centre_ind1, centre_ind2, width=62)

ctrclose = ctrdist <= spatial_threshold

allclosesti1 = ctrclose & ctr1closesti1 & ctr2closesti1
allclosesti2 = ctrclose & ctr1closesti2 & ctr2closesti2

ctrclose_expt12 = np.logical_xor(ctrclose, np.logical_or(allclosesti1,allclosesti2))
#%%
def ignore_shortevent(event, threshold=20):
    '''
    event: an array consists boolean value that indicates if an event happens
    threshold (ms): if the duration of consecutive events(length of consecutive True value) is below threshold then they will be deleted
    '''    
    '''
    find when two patterns in two different layers are spatially synchronized 
    (when the distance between the centre of two patterns are small enough) 
    sync: an array indicates when two patterns are synchronized;
    threshold: synchronization duration below threshold will be ignored
    '''
    dura = 0
    pre_true = 0
    onoff = np.zeros([2,1],dtype=int)
    on = 0 
    #off = 0
    for i in range(event.shape[0]):
        if event[i]:
            if pre_true == 0:
                pre_true = 1; dura += 1
                on = i
            else:
                dura += 1
        else:
            if pre_true == 1:
                pre_true = 0
                if dura < threshold:
                    event[i-dura:i] = False
                else:
                    onoff = np.concatenate((onoff, [[on],[i-1]]),1)
                dura = 0
            else:
                continue
    
    return event,onoff
#%%  

#%%          
allclosesti1_trim, onoff1 = ignore_shortevent(allclosesti1)  
allclosesti2_trim, onoff2 = ignore_shortevent(allclosesti2)     
ctrclose_expt12_trim, onoff3 = ignore_shortevent(ctrclose_expt12)                  

plt.figure(figsize=(10,6))
plt.plot(onoff1+starttime/ms, np.ones(onoff1.shape), 'b')
plt.plot(onoff2+starttime/ms, np.ones(onoff2.shape)*2, 'g')
plt.plot(onoff3+starttime/ms, np.ones(onoff3.shape)*3, 'r')
plt.yticks([0, 1, 2, 3, 4],['', 'sti1', 'sti2', 'other', ''])
plt.xticks(np.arange(0,6001,1000,dtype=int),['0', 'stimuli onset', '2000', '3000', '4000','5000','6000'])
plt.xlabel('time(ms)')
plt.plot([1000,1000],[0,4],'Black')
plt.title('Spatial synchronization of two patterns\nbottom-up: %.1f*default; top-down: %.1f*default\nstimuli onset:1000ms'%(scale_e_12[i], scale_e_21[j]))
#%%
plt.figure()
plt.plot(ctr1tosti1)
#%%
from scipy.fftpack import fft
#%%
mag = abs(fft(ctr1tosti1))
plt.figure()
plt.plot(mag)

#%%
t1 = np.arange(allclosesti1_trim.shape[0])[allclosesti1_trim]
t2 = np.arange(allclosesti2_trim.shape[0])[allclosesti2_trim]
t3 = np.arange(ctrclose_expt12_trim.shape[0])[ctrclose_expt12_trim]
    
allclosesti1_trim = allclosesti1_trim.astype(int)
allclosesti2_trim = allclosesti2_trim.astype(int)
ctrclose_expt12_trim = ctrclose_expt12_trim.astype(int)

allclosesti2_trim[allclosesti2_trim==1] = 2
ctrclose_expt12_trim[ctrclose_expt12_trim==1] = 3
#%%
plt.figure()
plt.scatter(t1, np.ones(t1.shape)*1)
plt.scatter(t2, np.ones(t2.shape)*2)
plt.scatter(t3, np.ones(t3.shape)*3)

#%%
plt.figure()
plt.plot(np.concatenate((np.arange([2]).reshape(1,-1),np.arange(3,5).reshape(1,-1)),0).T,np.ones([2,2]))
#plt.ylabel(['', 'sti1'])
plt.yticks([0, 1, 2],['', 'sti1', ''])


