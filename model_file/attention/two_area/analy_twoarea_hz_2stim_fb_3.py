#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:16:55 2021

@author: shni2598
"""


import matplotlib as mpl
mpl.use('Agg')
import scipy.stats
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fra
import frequency_analysis as fqa
import fano_mean_match
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
data_dir = 'raw_data/'
analy_type = 'wkfb'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly_fano10hz' #'data_anly' data_anly_temp

fftplot = 0; getfano = 1
get_nscorr = 0; get_nscorr_t = 0
get_TunningCurve = 0; get_HzTemp = 0
firing_rate_long = 0

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 0
else: get_ani = 0

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

# title = 'hz_1e2e%.2f_1e2i%.2f_2e1e%.2f_2e1i%.2f'%(data.inter.param.w_e1_e2_mean/5, data.inter.param.w_e1_i2_mean/5, \
#                                                     data.inter.param.w_e2_e1_mean/5, data.inter.param.w_e2_i1_mean/5)
if analy_type == 'fbrg': # fbrg: feedback range
    title = 'hz_2e1i%.2f_2e1er%.1f_2e1ir%.1f'%(data.inter.param.w_e2_i1_mean/5, \
                                               data.inter.param.tau_p_d_e2_e1, data.inter.param.tau_p_d_e2_i1)    
if analy_type == 'wkfb': # weak feedback
    title = 'hz_2e1e%.2f_2e1i%.2f'%(data.inter.param.w_e2_e1_mean/5, \
                                               data.inter.param.w_e2_i1_mean/5)    
    
    #%%
n_StimAmp = 4
n_perStimAmp = 50
stim_amp = 200*2**np.arange(n_StimAmp)
#%%
'''spontanous rate'''
dt = 1/10000;
end = int(20/dt); start = int(5/dt)
spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/15/data.a1.param.Ne
spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/15/data.a2.param.Ne

data_anly.spon_rate1 = spon_rate1
data_anly.spon_rate2 = spon_rate2
'''adapt rate'''
start = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0] - 2000)/1000/dt))
end = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0])/1000/dt))
adapt_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/2/data.a1.param.Ne
data_anly.adapt_rate1 = adapt_rate1
adapt_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/2/data.a2.param.Ne
data_anly.adapt_rate2 = adapt_rate2

title += '_1hz%.2f_1adphz%.2f_2hz%.2f_2adphz%.2f'%(spon_rate1,adapt_rate1,spon_rate2,adapt_rate2)

#%%
# def find_peakF(coef, freq, lwin):
#     dF = freq[1] - freq[0]
#     #Fwin = 0.3
#     #lwin = 3#int(Fwin/dF)
#     win = np.ones(lwin)/lwin
#     coef_avg = np.convolve(np.abs(coef[1:]), win, mode='same')
#     peakF = freq[1:][coef_avg.argmax()]
#     return peakF

# def plot_fft(freq, coef, freq_max1=20, freq_max2 = 200, fig=None, ax=None, label=''):
#     if fig is None:
#         fig, ax = plt.subplots(2,2,figsize=[9,9])
    
#     #fs = 1000
#     #data_fft = mua[:]
#     #coef, freq = fqa.myfft(data_fft, fs)
#     # data_anly.coef_spon = coef
#     # data_anly.freq_spon = freq
    
#     #peakF = find_peakF(coef, freq, 3)
    
#     #freq_max1 = 20
#     ind_len = freq[freq<freq_max1].shape[0] # int(20/(fs/2)data_fft*(len(data_fft)/2)) + 1
#     ax[0,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_linear')
#     ax[0,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_loglog')
    
#     #freq_max2 = 150
#     ind_len = freq[freq<freq_max2].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
#     ax[1,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_linear')
#     ax[1,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_loglog')
    
#     for i in range(2):
#         for j in range(2):
#             ax[i,j].legend()
    
#     return fig, ax, #peakF#, coef, freq

#%%
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%

start_time = 5e3; end_time = 20e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 5)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)


coef, freq = fqa.myfft(mua, Fs=1000)
peakF_spon = fqa.find_peakF(coef, freq, 3)
data_anly.coef_spon_a1 = coef
data_anly.freq_spon_a1 = freq
data_anly.peakF_spon_a1 =  peakF_spon
if fftplot and save_img:
    fig, ax = fqa.plot_fft(freq, coef, freq_max1=20, freq_max2 = 200, fig=None, ax=None, label='spon_a1')
    
    title1 = title + '_pf%.2f_a1'%(peakF_spon)
    savetitle = title1.replace('\n','')
    fig.suptitle(title1)
    fftfile = savetitle+'_fft_%d'%(loop_num)+'.png'
    fig.savefig(fftfile)
    plt.close()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 5)
mua = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

coef, freq = fqa.myfft(mua, Fs=1000)
peakF_spon = fqa.find_peakF(coef, freq, 3)
data_anly.coef_spon_a2 = coef
data_anly.freq_spon_a2 = freq
data_anly.peakF_spon_a2 =  peakF_spon

if fftplot and save_img:
    fig, ax = fqa.plot_fft(freq, coef, freq_max1=20, freq_max2 = 200, fig=None, ax=None, label='spon_a2')
    
    title1 = title + '_pf%.2f_a2'%(peakF_spon)
    savetitle = title1.replace('\n','')
    fig.suptitle(title1)
    fftfile = savetitle+'_fft_%d'%(loop_num)+'.png'
    fig.savefig(fftfile)
    plt.close()
    
start_time = data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0] - 2000
end_time = data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0]

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 5)
mua = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

coef, freq = fqa.myfft(mua, Fs=1000)
peakF_spon = fqa.find_peakF(coef, freq, 3)
data_anly.coef_adapt_a2 = coef
data_anly.freq_adapt_a2 = freq
data_anly.peakF_adapt_a2 =  peakF_spon


#%%
'''fano'''
if getfano:
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
    
    fanomm = fano_mean_match.fano_mean_match()
    # fanomm.bin_count_interval = 0.25
    bin_count_interval_hz = 10 # hz
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
        title3 = title + '_win%.1f_bin%d\n_range%d'%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
        fig.suptitle(title3)
        savetitle = title3.replace('\n','')
        fanofile = savetitle+'_%d'%(loop_num)+'.png'
        if save_img: fig.savefig(fanofile)
        plt.close()
#%%
'''noise correlation'''
if get_nscorr:
   
    nscorr = fra.noise_corr()
    
    neuron = np.arange(data.a1.param.Ne)
    neu_pool = [None]*2
    stim_loc = np.array([0,0])
    neu_range = 5
    dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
    neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]
    stim_loc = np.array([-32,-32])
    neu_range = 5
    dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
    neu_pool[1] = neuron[(dist >= 0) & (dist <= neu_range)]
        # neu_pool = [None]*1
        # neu_range = 5
        # neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]
    
    data_anly.nscorr = mydata.mydata()
    data_anly.nscorr.neu_pool = neu_pool
    
    simu_time_tot = data.param.simutime#29000
        
        #N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
    data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
    nscorr.spk_matrix1 = data.a1.ge.spk_matrix[neu_pool[0],:]
    nscorr.spk_matrix2 = data.a1.ge.spk_matrix[neu_pool[1],:]
    
    ns = np.zeros([2,n_StimAmp,2,2]); #[pair,n_StimAmp,att,mean-sem]
    
    #fig, ax = plt.subplots(1,1, figsize=[6,4])
    
    for st in range(n_StimAmp):  
        '''no-att; within 1 group'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
        nscorr.dura2 = None
        corr_mean, corr_sem = nscorr.get_nc_withingroup()
        ns[0, st, 0, 0], ns[0, st, 0, 1] = corr_mean, corr_sem
        '''att; within 1 group'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
        nscorr.dura2 = None
        corr_mean, corr_sem = nscorr.get_nc_withingroup()
        ns[0, st, 1, 0], ns[0, st, 1, 1] = corr_mean, corr_sem
        '''no-att; between 2 groups'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
        nscorr.dura2 = nscorr.dura1
        corr_mean, corr_sem = nscorr.get_nc_betweengroups()
        ns[1, st, 0, 0], ns[1, st, 0, 1] = corr_mean, corr_sem
        '''att; between 2 groups'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
        nscorr.dura2 = nscorr.dura1
        corr_mean, corr_sem = nscorr.get_nc_betweengroups()
        ns[1, st, 1, 0], ns[1, st, 1, 1] = corr_mean, corr_sem
    
    data_anly.nscorr.nscorr = ns
    
    fig, ax = plt.subplots(1,1, figsize=[6,4])
    ax.errorbar(np.arange(len(stim_amp)), ns[0,:,0,0], ns[0,:,0,1], c=clr[0], fmt='--', marker='o', label='no-att;1-group')
    ax.errorbar(np.arange(len(stim_amp)), ns[0,:,1,0], ns[0,:,1,1], c=clr[1], fmt='-', marker='o', label='att;1-group')
    ax.errorbar(np.arange(len(stim_amp)), ns[1,:,0,0], ns[1,:,0,1], c=clr[0], fmt='--', marker='x', label='no-att;2-group')
    ax.errorbar(np.arange(len(stim_amp)), ns[1,:,1,0], ns[1,:,1,1], c=clr[1], fmt='-', marker='x', label='att;2-group')

    #ax.legend()
    ax.legend()
    ax.xaxis.set_ticks([i for i in range(len(stim_amp))])
    ax.xaxis.set_ticklabels([str(item) for item in stim_amp])     
    #title3 = title + '_win%.1f_bin%d\n_range%d'%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    fig.suptitle(title)
    savetitle = title.replace('\n','')
    nsfile = savetitle+'_nc_%d'%(loop_num)+'.png'
    if save_img: fig.savefig(nsfile)
    plt.close()

#%%
'''noise correlation temporal'''
if get_nscorr_t:
    
    nscorr = fra.noise_corr()
    
    neuron = np.arange(data.a1.param.Ne)
    neu_pool = [None]*2
    stim_loc = np.array([0,0])
    neu_range = 5
    dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
    neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]
    stim_loc = np.array([-32,-32])
    neu_range = 5
    dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
    neu_pool[1] = neuron[(dist >= 0) & (dist <= neu_range)]
        # neu_pool = [None]*1
        # neu_range = 5
        # neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]
    
    data_anly.nscorr_t = mydata.mydata()
    data_anly.nscorr_t.neu_pool = neu_pool
    
    nscorr.win = 100 # ms sliding window length to count spikes
    nscorr.move_step = 20 # ms sliding window move step, (sampling interval for time varying noise correlation)
    nscorr.t_bf = -nscorr.win/2 # ms; time before stimulus onset to start to sample noise correlation
    nscorr.t_aft = -nscorr.win/2 # ms; time after stimulus off to finish sampling noise correlation
    data_anly.nscorr_t.param = data.class2dict(nscorr)
    
    simu_time_tot = data.param.simutime#29000
        
        #N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
    data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
    nscorr.spk_matrix1 = data.a1.ge.spk_matrix[neu_pool[0],:]
    nscorr.spk_matrix2 = data.a1.ge.spk_matrix[neu_pool[1],:]
    

    for st in range(n_StimAmp):  
        '''no-att; within 1 group'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
        #nscorr.dura2 = None
        corr_ = nscorr.get_nc_withingroup_t()
        if st == 0:
            ns_t = np.zeros([2,2,corr_.shape[1],n_StimAmp*2])
    
        ns_t[0, :, :, st] = corr_
        '''att; within 1 group'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
        #nscorr.dura2 = None
        corr_ = nscorr.get_nc_withingroup_t()
        ns_t[0, :, :, st+n_StimAmp] = corr_
        '''no-att; between 2 groups'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
        #nscorr.dura2 = nscorr.dura1
        corr_ = nscorr.get_nc_betweengroups_t()
        ns_t[1, :, :, st] = corr_
        '''att; between 2 groups'''
        nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
        #nscorr.dura2 = nscorr.dura1
        corr_ = nscorr.get_nc_betweengroups_t()
        ns_t[1, :, :, st+n_StimAmp] = corr_
    
    data_anly.nscorr_t.nscorr_t = ns_t
    fig, ax = plt.subplots(2,1, figsize=[8,6])
    sample_t = np.arange(ns_t.shape[2])*nscorr.move_step-nscorr.t_bf
    for st in range(n_StimAmp):  
        ax[0].errorbar(sample_t, ns_t[0, 0, :, st],ns_t[0, 1, :, st], c=clr[st], fmt='--', marker='o', label='no-att;1-group;amp:%.1fHz'%stim_amp[st])
        ax[0].errorbar(sample_t, ns_t[0, 0, :, st+n_StimAmp],ns_t[0, 1, :, st+n_StimAmp], c=clr[st], fmt='-', marker='o', label='att;1-group;amp:%.1fHz'%stim_amp[st])
        ax[1].errorbar(sample_t, ns_t[1, 0, :, st],ns_t[1, 1, :, st], c=clr[st], fmt='--', marker='x', label='no-att;2-group;amp:%.1fHz'%stim_amp[st])
        ax[1].errorbar(sample_t, ns_t[1, 0, :, st+n_StimAmp],ns_t[1, 1, :, st+n_StimAmp], c=clr[st], fmt='-', marker='x', label='att;2-group;amp:%.1fHz'%stim_amp[st])
    
    #ax.legend()
    ax[0].legend()
    ax[0].set_xlim([sample_t.min()-20,sample_t.max()+150])
    ax[1].legend()
    ax[1].set_xlim([sample_t.min()-20,sample_t.max()+150])
    fig.suptitle(title)
    savetitle = title.replace('\n','')
    nsfile = savetitle+'_nct_%d'%(loop_num)+'.png'
    if save_img: fig.savefig(nsfile)
    plt.close()
#%%
'''firing rate'''
'''
firing rate-location ; tunning curve
'''
if get_TunningCurve:
    stim_loc = np.array([0,0])
    
    dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
    dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
    n_in_bin = [None]*(dist_bin.shape[0]-1)
    neuron = np.arange(data.a1.param.Ne)
    
    for i in range(len(dist_bin)-1):
        n_in_bin[i] = neuron[(dist >= dist_bin[i]) & (dist < dist_bin[i+1])]
    
    
    simu_time_tot = data.param.simutime#29000
    
    #N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
    N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
    
    data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
    ## stim
    hz_loc = fra.tuning_curve(data.a1.ge.spk_matrix, data.a1.param.stim1.stim_on, n_in_bin)
    ## spon
    hz_loc_spon = np.zeros([2, len(n_in_bin)])
    '''no attention'''
    spon_onff = np.array([[5000,20000]])
    hz_loc_spon[0,:] = fra.tuning_curve(data.a1.ge.spk_matrix, spon_onff, n_in_bin)
    
    '''attention'''
    spon_adpt_stt = data.a1.param.stim1.stim_on[N_stim,0] - 2000 
    spon_adpt_end = data.a1.param.stim1.stim_on[N_stim,0]
    spon_onff = np.array([[spon_adpt_stt,spon_adpt_end]])
    hz_loc_spon[1,:] = fra.tuning_curve(data.a1.ge.spk_matrix, spon_onff, n_in_bin)
    
    data_anly.hz_loc = hz_loc.copy()
    data_anly.hz_loc_spon = hz_loc_spon.copy()
    
    '''plot'''
    # n_StimAmp = 4
    # n_perStimAmp = 50
    # stim_amp = 200*2**np.arange(n_StimAmp)
    plt.figure(figsize=[8,6])
    for i in range(n_StimAmp):
        
        hz_loc_mean_noatt = hz_loc[i*n_perStimAmp:(i+1)*n_perStimAmp,:].mean(0)
        hz_loc_mean_att = hz_loc[(i+n_StimAmp)*n_perStimAmp:(i+n_StimAmp+1)*n_perStimAmp,:].mean(0)
        
        
        dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]
            
        plt.plot(dist_bin_plot, hz_loc_mean_noatt, ls='--', marker='o', c=clr[i], label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
        
        plt.plot(dist_bin_plot, hz_loc_mean_att, ls='-', marker='o', c=clr[i], label = 'attention; stim_amp: %.1f Hz'%(stim_amp[i]))

    plt.plot(dist_bin_plot, hz_loc_spon[0], ls='--', marker='o', c=clr[i+1], label = 'spontaneous')
    plt.plot(dist_bin_plot, hz_loc_spon[1], ls='-', marker='o', c=clr[i+1], label = 'attention; spontaneous')
    
    
    #title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
    #plt.title('average firing rate versus distance to stimulus centre; 2e1e:%.3f 2e1i%.3f\n Spike-count bin: %.1f ms\n'%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]],fr_bin[b])+title)
    #plt.title('average firing rate versus distance to stimulus centre; Spike-count bin: %.1f ms\n'%(data.a1.param.stim1.stim_on[0,1]-data.a1.param.stim1.stim_on[0,0]))#+title)
    plt.title(title + '\nSpike-count bin: %.1f ms\n'%(data.a1.param.stim1.stim_on[0,1]-data.a1.param.stim1.stim_on[0,0]))#+title)
    
    plt.xlim([dist_bin[0],dist_bin[-1]])
    plt.xlabel('distance')
    plt.ylabel('Hz')
    plt.legend()
    if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
    plt.close()
    
    del hz_loc, hz_loc_spon
#%%
'''firing rate time'''
if get_HzTemp:
    
    # n_StimAmp = 4
    # n_perStimAmp = 50
    # stim_amp = 200*2**np.arange(n_StimAmp)

    mua_loc_ind = -1
    fig, ax, = plt.subplots(2,1, figsize=[8,10])
    for mua_loca_1 in [[0,0]]:#,[-32,-32]]:
        mua_loc_ind += 1
        N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
        #stim_amp = 400
        #mua_loca_1 = [0, 0]
        mua_range_1 = 5
        mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
        window = 5
        dura_onoff = data.a1.param.stim1.stim_on.copy()
        dura_onoff[:,0] -= 100
        dura_onoff[:,1] += 100
        hz_t = fra.firing_rate_time_multi(data.a1.ge, mua_neuron_1, dura_onoff, window=window)
        data_anly.hz_t = hz_t.copy()
        
        hz_t = hz_t.reshape(n_StimAmp*2, n_perStimAmp, -1)
        
        #hz_t_mean = hz_t.mean(1)
        for i in range(n_StimAmp):
            hz_t_noatt_mean = hz_t[i].mean(0)
            #hz_t_noatt_std = hz_t[:N_stim,:].std(0)
            hz_t_noatt_sem = scipy.stats.sem(hz_t[i], 0, nan_policy='omit')
            
            hz_t_att_mean = hz_t[i+n_StimAmp].mean(0)
            #hz_t_att_std = hz_t[N_stim:N_stim*2,:].std(0)
            hz_t_att_sem = scipy.stats.sem(hz_t[i+n_StimAmp], 0, nan_policy='omit')
            
            
            #hz_t_mean = hz_t[:,:, 0].reshape(n_amp_stim,n_per_amp,-1).mean(1)
            #plt.figure(figsize=[8,6])
            #for i in range(n_amp_stim_att):
            t_plot = np.arange(dura_onoff[0,1] - dura_onoff[0,0]) - 100
            ax[0].plot(t_plot, hz_t_noatt_mean, ls='--', c=clr[i], label = 'stim_amp: %.1f Hz;\nloc: [%.1f,%.1f]'%(stim_amp[i],mua_loca_1[0],mua_loca_1[1]))
            ax[0].fill_between(t_plot, hz_t_noatt_mean-hz_t_noatt_sem, hz_t_noatt_mean+hz_t_noatt_sem, \
                             ls='--', facecolor=clr[i], edgecolor=clr[i], alpha=0.2)
            #for i in range(n_amp_stim_att,n_amp_stim):
            ax[0].plot(t_plot, hz_t_att_mean, ls='-', c=clr[i], label = 'att; stim_amp: %.1f Hz;\nloc: [%.1f,%.1f]'%(stim_amp[i],mua_loca_1[0],mua_loca_1[1])) 
            ax[0].fill_between(t_plot, hz_t_att_mean-hz_t_att_sem, hz_t_att_mean+hz_t_att_sem, \
                             ls='-', facecolor=clr[i], edgecolor=clr[i], alpha=0.2)
    
    ax[0].set_xlim([t_plot[0]+50, t_plot[-1]+100])
    ax[0].set_xlabel('ms')
    ax[0].set_ylabel('Hz')
    #title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
    #title = ''
    ax[0].set_title(title + '; senssory')
    ax[0].legend()
    del hz_t
    
    n_in_bin = []
    mua_loca = np.array([0,0])
    mua_range = 5
    mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
    n_in_bin.append(mua_neuron)
    
    mua_loca = np.array([-10,0])
    mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
    n_in_bin.append(mua_neuron)
   
    mua_loca = [[0,0],[-10,0]]
    simu_time_tot = data.param.simutime#29000
    
    #N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
    N_stim = int(round(data.a1.param.stim1.stim_amp_scale.shape[0]/2))
    
    data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
    ## stim
    hz_loc_elec = fra.tuning_curve(data.a1.ge.spk_matrix, data.a1.param.stim1.stim_on, n_in_bin)
    
    data_anly.hz_loc_elec = hz_loc_elec.copy()
    
    hz_loc_elec = hz_loc_elec.reshape(n_StimAmp*2, n_perStimAmp, -1)
    
    hz_loc_elec_mean = hz_loc_elec.mean(1)
    hz_loc_elec_sem = scipy.stats.sem(hz_loc_elec, 1, nan_policy='omit')
    '''no-stim; spon; no attention'''
    spon_onff = np.array([[5000,20000]])
    hz_loc_spon = np.zeros([2,2])
    hz_loc_spon[0,:] = fra.tuning_curve(data.a1.ge.spk_matrix, spon_onff, n_in_bin)
    
    '''no-stim; spon; attention'''
    spon_adpt_stt = data.a1.param.stim1.stim_on[N_stim,0] - 2000 
    spon_adpt_end = data.a1.param.stim1.stim_on[N_stim,0]
    spon_onff = np.array([[spon_adpt_stt,spon_adpt_end]])
    hz_loc_spon[1,:] = fra.tuning_curve(data.a1.ge.spk_matrix, spon_onff, n_in_bin)
    
    hz_loc_elec_mean_tmp = np.zeros([n_StimAmp*2+2, 2])
    hz_loc_elec_mean_tmp[1:n_StimAmp+1] = hz_loc_elec_mean[:n_StimAmp]
    hz_loc_elec_mean_tmp[n_StimAmp+2:] = hz_loc_elec_mean[n_StimAmp:]
    hz_loc_elec_mean_tmp[0] = hz_loc_spon[0]
    hz_loc_elec_mean_tmp[n_StimAmp+1] = hz_loc_spon[1]
    hz_loc_elec_mean = hz_loc_elec_mean_tmp
    
    hz_loc_elec_sem_tmp = np.zeros([n_StimAmp*2+2, 2])
    hz_loc_elec_sem_tmp[1:n_StimAmp+1] = hz_loc_elec_sem[:n_StimAmp]
    hz_loc_elec_sem_tmp[n_StimAmp+2:] = hz_loc_elec_sem[n_StimAmp:]
    hz_loc_elec_sem = hz_loc_elec_sem_tmp    
    
    data_anly.hz_loc_elec_mean = hz_loc_elec_mean.copy()
    data_anly.hz_loc_elec_sem = hz_loc_elec_sem.copy()
    
    stim_amp_new = np.concatenate(([0],stim_amp))
    for i in range(len(n_in_bin)):
        ax[1].errorbar(np.arange(stim_amp_new.shape[0]), hz_loc_elec_mean[:n_StimAmp+1,i], hz_loc_elec_sem[:n_StimAmp+1,i], \
                     fmt='--', c=clr[i], marker='o', label='no attention, loc: [%.1f,%.1f]'%(mua_loca[i][0],mua_loca[i][1]))
        ax[1].errorbar(np.arange(stim_amp_new.shape[0]), hz_loc_elec_mean[n_StimAmp+1:2*n_StimAmp+2,i], hz_loc_elec_sem[n_StimAmp+1:2*n_StimAmp+2,i], \
                     fmt='-', c=clr[i], marker='o', label='attention, loc: [%.1f,%.1f]'%(mua_loca[i][0],mua_loca[i][1]))
    ax[1].set_ylabel('hz')
    
    ax_pct = ax[1].twinx()
    ax_pct.set_ylabel('rate increase percentage')
    for i in range(len(n_in_bin)):
        inc_percent = (hz_loc_elec_mean[n_StimAmp+1:2*n_StimAmp+2,i] - hz_loc_elec_mean[:n_StimAmp+1,i])/hz_loc_elec_mean[:n_StimAmp+1,i]*100
        ax_pct.plot(np.arange(stim_amp_new.shape[0]), inc_percent, ls='-.', c=clr[i], label='loc: [%.1f,%.1f]'%(mua_loca[i][0],mua_loca[i][1]))
    
    ax_pct.legend()
    ax[1].legend()
    ax[1].xaxis.set_ticks([i for i in range(len(stim_amp_new))])
    ax[1].xaxis.set_ticklabels([str(item) for item in stim_amp_new])     
    
    #plt.savefig(save_dir+title.replace(':','')+'_temp'+'_%d'%ie_ind+'.png')
    if save_img: fig.savefig(title.replace('\n','')+'_temp'+'_%d'%loop_num+'.png')
    plt.close()
    
    del hz_loc_elec
#%%
# fig, ax = plt.subplots(1,1)
# plt.errorbar(np.arange(3),np.arange(3), np.arange(3),fmt='--',marker='o',c='r')
# ax.plot(arange(4))
# ax.xaxis.set_ticks([i for i in range(len(stim_amp))])
# ax.xaxis.set_ticklabels([str(item) for item in stim_amp])
# #ax.get_xticklabels()   
#%%
if firing_rate_long:
    '''firing rate long temp'''
    mua_range_1 = 5
    mua_loca_1 = [0,0]
    mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)
    
    mua_range_2 = 5
    mua_loca_2 = [0,0]
    mua_neuron_2 = cn.findnearbyneuron.findnearbyneuron(data.a2.param.e_lattice, mua_loca_2, mua_range_2, data.a2.param.width)
    window = 10
    '''attention'''
    sti_on = 4000; adpt_on = 2000 
    start_time = data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0] - sti_on
    end_time = data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp+5,0] + 500 
    # fra.get_spike_rate(data.a1.ge, start_time, end_time, indiv_rate = True, popu_rate = False, \
    #                    sample_interval = 1, n_neuron = data.a1.param.Ne, window = 10, dt = 0.1, reshape_indiv_rate = False, save_results_to_input = False):
    data.a1.ge.get_spike_rate(start_time,end_time,window=window,n_neuron = data.a1.param.Ne,reshape_indiv_rate = False)
    mua_1_adapt = data.a1.ge.spk_rate.spk_rate[mua_neuron_1].mean(0)/(window/1000)
    data.a2.ge.get_spike_rate(start_time,end_time,window=window,n_neuron = data.a2.param.Ne,reshape_indiv_rate = False)
    mua_2_adapt = data.a2.ge.spk_rate.spk_rate[mua_neuron_2].mean(0)/(window/1000)
    '''no attention'''
    start_time = data.a1.param.stim1.stim_on[0,0] - sti_on
    end_time = data.a1.param.stim1.stim_on[5,0] + 500 
    # fra.get_spike_rate(data.a1.ge, start_time, end_time, indiv_rate = True, popu_rate = False, \
    #                    sample_interval = 1, n_neuron = data.a1.param.Ne, window = 10, dt = 0.1, reshape_indiv_rate = False, save_results_to_input = False):
    data.a1.ge.get_spike_rate(start_time,end_time,window=window,n_neuron = data.a1.param.Ne,reshape_indiv_rate = False)
    mua_1 = data.a1.ge.spk_rate.spk_rate[mua_neuron_1].mean(0)/(window/1000)
    data.a2.ge.get_spike_rate(start_time,end_time,window=window,n_neuron = data.a2.param.Ne,reshape_indiv_rate = False)
    mua_2 = data.a2.ge.spk_rate.spk_rate[mua_neuron_2].mean(0)/(window/1000)
    
    data_list = [mua_1, mua_2, mua_1_adapt, mua_2_adapt]
    fig, ax = plt.subplots(4,1,figsize=[8,6])
    ax[0].plot(mua_1, label='sens; no-att')
    ax[1].plot(mua_2, label='asso; no-att')
    ax[2].plot(mua_1_adapt, label='sens; att')
    ax[3].plot(mua_2_adapt, label='asso; att')
    for axi in range(len(ax)):
        ax[axi].plot([sti_on,sti_on],[0,data_list[axi].max()], '--r',label='stim on')
        ax[axi].set_xlim([ax[axi].get_xlim()[0],ax[axi].get_xlim()[1]+1700])
        ax[axi].legend()
        if axi%2 == 1:
            ax[axi].plot([adpt_on,adpt_on],[0,data_list[axi].max()], '--m',label='adapt on')
            ax[axi].legend()
    
        fig.suptitle(title)
        #savetitle = title.replace('\n','')
        #nsfile = savetitle+'_nct_%d'%(loop_num)+'.png'
        fig.savefig(title.replace('\n','')+'_fr_%d'%(loop_num)+'.png')
        plt.close()
#%%
'''animation'''
if get_ani:
    '''spon'''
    first_stim = 0; last_stim = 0
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 2000
    end_time = data.a1.param.stim1.stim_on[last_stim,0] 
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    # stim_on_off = data.a1.param.stim1.stim_on-start_time
    # stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    # #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    # stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    # adpt = None
    # #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    # #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=None, adpt=None)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_spon_%d'%loop_num+'.mp4'
    
    ani.save(moviefile)
    del ani
    
    first_stim = 49; last_stim = 50
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = data.a1.param.stim1.stim_on[last_stim,0] + 500
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    stim_on_off = data.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    adpt = None
    #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_noatt_%d'%loop_num+'.mp4'
    
    # if loop_num%1 == 0:
    ani.save(moviefile)
    del ani
    
    first_stim = 149; last_stim = 150
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = data.a1.param.stim1.stim_on[last_stim,0] + 500
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    stim_on_off = data.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    adpt = None
    #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_noatt_hipt_%d'%loop_num+'.mp4'
    
    # if loop_num%1 == 0:
    ani.save(moviefile)
    del ani
        # pass
    
    first_stim = 249; last_stim = 250
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = data.a1.param.stim1.stim_on[last_stim,0] + 500
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    stim_on_off = data.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    #stim = [[[[31.5,31.5],[-0.5,63.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
    #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]

    adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    #adpt = None
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_att_%d'%loop_num+'.mp4'
    
    # if loop_num%1 == 0:
    ani.save(moviefile)
    del ani
    #     pass
#%%
data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)
#data_anly.save(data_anly.class2dict(), datapath+'data_anly_fano%d.file'%loop_num)
#data_anly.save(data_anly.class2dict(), datapath+'data_anly_fano5hz%d.file'%loop_num)
