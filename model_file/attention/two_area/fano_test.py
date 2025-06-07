#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:19:31 2021

@author: shni2598
"""

'''fano'''
#if getfano:
data_anly.fano = mydata.mydata()

stim_loc = np.array([0,0])

neuron = np.arange(data.a1.param.Ne)

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool = [None]*1
neu_pool[0] = neuron[(dist >= 0) & (dist <= 10)]

#fr_bin = np.array([200]) 
    
simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

fanomm = fano_mean_match.fano_mean_match()
# fanomm.bin_count_interval = 0.25
bin_count_interval_hz = 5 # hz
fanomm.spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]
fanomm.method = 'regression' # 'mean' or 'regression'
fanomm.mean_match_across_condition = True # if do mean matching across different condition e.g. attention or no-attention condition
fanomm.seed = 100


N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
win_lst = [50, 100]
data_anly.fano.win_lst = win_lst
data_anly.fano.bin_count_interval_hz = bin_count_interval_hz
win_id = -1
for win in win_lst:#[50,100,150]:
    win_id += 1
    for st in range(n_StimAmp):
        fanomm.bin_count_interval = win*10**-3*bin_count_interval_hz
        fanomm.win = win
        fanomm.stim_onoff = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
        fanomm.stim_onoff_2 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy() #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True
        fanomm.t_bf = -(win/2)
        fanomm.t_aft = -(win/2)
        #fano_mean_noatt, fano_std_noatt, _ = fanomm.get_fano()
        fano_mean_noatt, fano_sem_noatt, _, fano_mean_att, fano_sem_att, _ = fanomm.get_fano()
        
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