#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:52:27 2021

@author: shni2598
"""

import brian2.numpy_ as np
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import connection as cn
#import warnings
from scipy import sparse
import fano_mean_match
#%%
lam1 = [8, 16, 2]; #[8, 16, 4] [8, 16, 2]
lam2 = [3, 1.5, 1];
scales1 = [1, 0.5, 4] # [1, 0.5, 2] [1, 0.5, 4]
scales2 = [1, 2, 3]
block = [[6,3],[5,4],[2,7]] #[[6,3],[5,4],[3,6]] [[6,3],[5,4],[2,7]]

win = 50;
t = np.arange(win)

trial = 100
mat_list = []

mat_ = np.zeros([9,150])

for tri in range(trial):
    t_ind = []
    i_ind = []
    for blc in range(len(block)):
        for m in range(len(block[blc])):
            if m == 0:
                for i in range(block[blc][m]):
                    t_c = win*blc + np.random.choice(t, int(round(scales1[blc]*np.random.poisson(lam1[blc]))), replace=False)
                    t_ind.append(t_c)
                    i_ind.append(np.ones(len(t_c), int)*i)
            elif m == 1:
                for i in range(block[blc][m-1], block[blc][m-1]+block[blc][m]):
                    t_c = win*blc + np.random.choice(t, int(round(scales2[blc]*np.random.poisson(lam2[blc]))), replace=False)
                    t_ind.append(t_c)
                    i_ind.append(np.ones(len(t_c), int)*i)
            else: pass
    mat_[:] = 0 # = np.zeros([9,150])
    mat_[np.concatenate(i_ind), np.concatenate(t_ind)] = 1
    mat_list.append(mat_.copy())
#%%
mat_1 = np.concatenate(mat_list, 1)
mat_2 = np.concatenate(mat_list, 1)
mat = np.concatenate((mat_1, mat_2), 1)
mat_sparse = sparse.csc_matrix(mat)


#%%


            
            
    
#     for i in range(6):
#         for blc in range(len(scales1)):            
#             t_c = win*blc + np.random.choice(t, int(round(scales1[blc]*np.random.poisson(lam1))), replace=False)
#             t_ind.append(t_c)
#             i_ind.append(np.ones(len(t_c), int)*i)
    
#     for i in range(6,9):
#         for blc in range(len(scales2)):            
#             t_c = win*blc + np.random.choice(t, int(round(scales2[blc]*np.random.poisson(lam2))), replace=False)
#             t_ind.append(t_c)
#             i_ind.append(np.ones(len(t_c), int)*i)
            
#     mat_[:] = 0 # = np.zeros([9,150])
#     mat_[np.concatenate(i_ind), np.concatenate(t_ind)] = 1
#     mat_list.append(mat_.copy())
# #%%
# mat = np.concatenate(mat_list, 1)
# mat_sparse = sparse.csc_matrix(mat)

#%%
plt.figure()
plt.imshow(mat, aspect='auto')
#%%
plt.figure()
plt.plot(mat_sparse.nonzero()[1],mat_sparse.nonzero()[0], '.')
#%%
dura_on = np.zeros([trial,2])
dura_on[:,0] = np.arange(trial)*150
dura_on[:,1] = np.arange(1,trial+1)*150
#%%
dura_on_2 = np.zeros([trial,2])
dura_on_2[:,0] = np.arange(trial)*150 + 15000
dura_on_2[:,1] = np.arange(1,trial+1)*150 + 15000
#%%

data_anly.fano = mydata.mydata()

stim_loc = np.array([0,0])

neuron = np.arange(data.a1.param.Ne)

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool = [None]*1
neu_range = 5
neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]
data_anly.fano.neu_range = neu_range
#fr_bin = np.array([200]) 
    
simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
#%%
fanomm = fano_mean_match.fano_mean_match()
# fanomm.bin_count_interval = 0.25
bin_count_interval_hz = 44 # hz
fanomm.spk_sparmat = mat_sparse #data.a1.ge.spk_matrix[neu_pool[0],:]
fanomm.method = 'mean' # 'mean' or 'regression'
fanomm.mean_match_across_condition = True # if do mean matching across different condition e.g. attention or no-attention condition
fanomm.seed = 100
fanomm.dt = 1
fanomm.move_step = 50

#N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
win_lst = [50]
#data_anly.fano.win_lst = win_lst
#data_anly.fano.bin_count_interval_hz = bin_count_interval_hz
win_id = -1
for win in win_lst:#[50,100,150]:
    win_id += 1
    for st in range(1):
        fanomm.bin_count_interval = win*10**-3*bin_count_interval_hz
        fanomm.win = win
        fanomm.stim_onoff = dura_on #data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
        fanomm.stim_onoff_2 = dura_on_2
        #fanomm.stim_onoff_2 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy() #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True
        fanomm.t_bf = -(win/2)
        fanomm.t_aft = -(win/2)
        #fano_mean_noatt, fano_std_noatt, _ = fanomm.get_fano()
        #fano_mean_noatt, fano_sem_noatt, _, fano_mean_att, fano_sem_att, _ = fanomm.get_fano()
        
        #fano_mean, fano_sem, fano_record = fanomm.get_fano()
        fano_mean_noatt, fano_sem_noatt, _, fano_mean_att, fano_sem_att, _ = fanomm.get_fano()

#%%
fig, ax = plt.subplots(1,1, figsize=[8,6])
#for st in range(n_StimAmp):
    # ax.errorbar(np.arange(data_anly.fano.fano_mean_sem[win_id].shape[1])*10+(win/2), \
    #             data_anly.fano.fano_mean_sem[win_id][st,:,0],data_anly.fano.fano_mean_sem[win_id][st,:,1], \

ax.errorbar(np.arange(fano_mean_noatt.shape[0]), fano_mean_noatt, fano_sem_noatt)
ax.errorbar(np.arange(fano_mean_att.shape[0]), fano_mean_att, fano_sem_att)



#%%
        if win_id ==0 and st == 0:
            data_anly.fano.fano_mean_sem = [None]*len(win_lst)#np.zeros([n_StimAmp*2, fano_mean_noatt.shape[0], 2, len(win_lst)])
        if st == 0:
            data_anly.fano.fano_mean_sem[win_id] = np.zeros([n_StimAmp*2, fano_mean_noatt.shape[0], 2])
        data_anly.fano.fano_mean_sem[win_id][st,:,0] = fano_mean_noatt
        data_anly.fano.fano_mean_sem[win_id][st+n_StimAmp,:,0] = fano_mean_att
        data_anly.fano.fano_mean_sem[win_id][st,:,1] = fano_sem_noatt
        data_anly.fano.fano_mean_sem[win_id][st+n_StimAmp,:,1] = fano_sem_att
        
    # data_anly.fano.fano_mean_noatt = fano_mean_noatt
    # data_anly.fano.fano_mean_att = fano_mean_att
    # data_anly.fano.fano_sem_att = fano_sem_att        
    # data_anly.fano.fano_sem_noatt = fano_sem_noatt
    
    # print(np.sum(np.isnan(_)))
    # print(np.sum(np.isnan(fano_mean_noatt)))
    # print(np.sum(np.isnan(fano_std_noatt)))
    # fanomm.stim_onoff = data.a1.param.stim1.stim_on[N_stim:N_stim*2].copy()
    
    # fano_mean_att, fano_std_att, _ = fanomm.get_fano()
    # # print(np.sum(np.isnan(_)))
    # data_anly.fano.fano_mean_att = fano_mean_att
    # data_anly.fano.fano_std_att = fano_std_att
    
    fig, ax = plt.subplots(1,1, figsize=[8,6])
    for st in range(n_StimAmp):
        ax.errorbar(np.arange(data_anly.fano.fano_mean_sem[win_id].shape[1])*10+(win/2), \
                    data_anly.fano.fano_mean_sem[win_id][st,:,0],data_anly.fano.fano_mean_sem[win_id][st,:,1], \
                    fmt='--', c=clr[st], marker='o', label='no att, stim_amp: %.1f Hz'%(stim_amp[st]))
        ax.errorbar(np.arange(data_anly.fano.fano_mean_sem[win_id].shape[1])*10+(win/2),\
                    data_anly.fano.fano_mean_sem[win_id][st+n_StimAmp,:,0],data_anly.fano.fano_mean_sem[win_id][st+n_StimAmp,:,1],\
                    fmt='-', c=clr[st], marker='o', label='att, stim_amp: %.1f Hz'%(stim_amp[st]))
    ax.set_xlabel('ms')
    ax.set_ylabel('fano')
    plt.legend()
    title3 = title + '_win%.1f_bin%d\n_range%d'%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    fig.suptitle(title3)
    savetitle = title3.replace('\n','')
    fanofile = savetitle+'_%d'%(loop_num)+'.png'






#%%

np.random.poisson(lam1)








 