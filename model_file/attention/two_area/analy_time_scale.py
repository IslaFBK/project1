#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:11:15 2021

@author: shni2598
"""



import matplotlib as mpl
mpl.use('Agg')
from scipy.stats import sem
from scipy.optimize import curve_fit
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
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/fb/movie1/' + data_dir
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly' #'data_anly' data_anly_temp

fftplot = 0; getfano = 0
get_nscorr = 0; get_nscorr_t = 0
get_TunningCurve = 0; get_HzTemp = 0; get_HzTemp2 = 0
firing_rate_long = 0

save_img = 1

get_ani = 1
# if loop_num%4 == 0: save_img = 1
# else: save_img = 0

# if loop_num%10 ==0: get_ani = 0
# else: get_ani = 0

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
'''spontanous rate'''
dt = 1/10000;
end = int(7/dt); start = int(5/dt)
spon_rate1 = 0
spon_rate2 = 0
for tr in range(100):
    spon_rate1 += np.sum((data.a1.ge[tr]['t'] < end) & (data.a1.ge[tr]['t'] >= start))/2/data.a1.param.Ne
    spon_rate2 += np.sum((data.a2.ge[tr]['t'] < end) & (data.a2.ge[tr]['t'] >= start))/2/data.a2.param.Ne
spon_rate1 /= 100
spon_rate2 /= 100
#spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/15/data.a2.param.Ne
title += '_1hz%.2f_2hz%.2f'%(spon_rate1,spon_rate2)

#%%
def time_scale(sua_trial, delay_max=6):
    coef = np.zeros([sua_trial.shape[0], sua_trial.shape[1], sua_trial.shape[1]])
    for i in range(sua_trial.shape[0]):
        coef[i] = np.corrcoef(sua_trial[i])
    coef_tau = np.zeros([2, delay_max+1])
    for i in range(0,delay_max+1):
        coef_tau[0, i] = np.nanmean(coef.diagonal(i,1,2))
        coef_tau[1, i] = sem(coef.diagonal(i,1,2).reshape(-1),  nan_policy='omit')
    return coef_tau, coef
#%%
data_tr = mydata.mydata()
data_tr.a1 = mydata.mydata()
data_tr.a1.ge = mydata.mydata()
data_tr.a1.gi = mydata.mydata()
data_tr.a2 = mydata.mydata()
data_tr.a2.ge = mydata.mydata()
data_tr.a2.gi = mydata.mydata()



start_time = 5e3; end_time = 6200
window = 50
sample_interval = 5
# data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=50, n_neuron = data.a1.param.Ne, window = window)
# data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=50, n_neuron = data.a2.param.Ne, window = window)



trial_num = 100
for tr in range(trial_num):
    data_tr.a1.ge.i = data.a1.ge[tr]['i']
    data_tr.a1.ge.t = data.a1.ge[tr]['t']
    data_tr.a2.ge.i = data.a2.ge[tr]['i']
    data_tr.a2.ge.t = data.a2.ge[tr]['t']
    
    data_tr.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=sample_interval, n_neuron = data.a1.param.Ne, window = window)
    data_tr.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=sample_interval, n_neuron = data.a2.param.Ne, window = window)
    if tr == 0:
        sua1 = np.zeros([data.a1.param.Ne, data_tr.a1.ge.spk_rate.spk_rate.shape[-1], trial_num])
        sua2 = np.zeros(sua1.shape)
    
    sua1[:,:,tr] = data_tr.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1).astype(int)
    sua2[:,:,tr] = data_tr.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1).astype(int)

t_max = 1000 # ms

coef_tau1, coef1 = time_scale(sua1, delay_max=int(round(t_max/sample_interval)))
coef_tau2, coef2 = time_scale(sua2, delay_max=int(round(t_max/sample_interval)))

data_anly.time_scale = mydata.mydata()
data_anly.time_scale.coef_tau1 = coef_tau1
data_anly.time_scale.coef_tau2 = coef_tau2

#%%
# def damp_oscil(x, y0_1, tau1, f1):
#     return y0_1 * np.exp(-x/tau1) * np.cos(2*np.pi*f1*x + 0)
def damp_oscil(x, tau1, f1):
    return 1 * np.exp(-x/tau1) * np.cos(2*np.pi*f1*x + 0)        
# def damp_oscil(x, y0_1, tau1, f1, y0_2, tau2, f2):
#     return y0_1 * np.exp(-x/tau1) * np.cos(2*np.pi*f1*x + 0) + \
#         y0_2 * np.exp(-x/tau2) * np.cos(2*np.pi*f2*x + 0)

# def damp_oscil(x, y0_1, tau1, f1, y0_2, tau2):
#     return (y0_1 * np.exp(-x/tau1) + y0_2 * np.exp(-x/tau2)) * np.cos(2*np.pi*f1*x + 0)
        
        
t_delay_1 = np.arange(coef_tau1[0].shape[0])*sample_interval
t_delay_2 = np.arange(coef_tau2[0].shape[0])*sample_interval

#f = 2; tau = 5; phi = 0
#x = np.linspace(0,20,200)
popt1, pcov1 = curve_fit(damp_oscil, t_delay_1/1000, coef_tau1[0], p0=[0.3, 3])
popt2, pcov2 = curve_fit(damp_oscil, t_delay_2/1000, coef_tau2[0], p0=[0.3, 3])
data_anly.time_scale.popt1 = popt1
data_anly.time_scale.pcov1 = pcov1
data_anly.time_scale.popt2 = popt2
data_anly.time_scale.pcov2 = pcov2
#%%
fig, ax = plt.subplots(1,1)

t_delay_1 = np.arange(coef_tau1[0].shape[0])*sample_interval
t_delay_2 = np.arange(coef_tau2[0].shape[0])*sample_interval

ax.errorbar(t_delay_1, coef_tau1[0], coef_tau1[1], \
                     fmt='--', c=clr[0], marker='.', label='sens')
ax.errorbar(t_delay_2, coef_tau2[0], coef_tau2[1], \
                     fmt='--', c=clr[1], marker='.', label='asso')

ax.plot(t_delay_1, damp_oscil(t_delay_1/1000, *popt1), \
        c=clr[0], label='exp(-x/tau)*np.cos(2*pi*f*x);tau:%.2f,f:%.2f'%(popt1[0]*1000, popt1[1]))  
ax.plot(t_delay_2, damp_oscil(t_delay_2/1000, *popt2), \
        c=clr[1], label='exp(-x/tau)*np.cos(2*pi*f*x);tau:%.2f,f:%.2f'%(popt2[0]*1000, popt2[1]))    

# ax.plot(t_delay_1, np.abs(hb1), \
#         c=clr[0], label='sens-fit')  
# ax.plot(t_delay_2, np.abs(hb2), \
#         c=clr[1], label='asso-fit')  
ax.legend()    
#time_shift = np.arange(1, coef_tau1.shape[1]+1)*sample_interval #+ window/2
#time_shift = np.arange(1,11)*window
#ax.xaxis.set_ticks([i for i in range(len(time_shift))])
#ax.xaxis.set_ticklabels([str(item) for item in time_shift]) 
ax.set_xlabel('ms') 
ax.set_title(title + '\nauto-correlation')
fig.savefig(title+'_acr%d.png'%loop_num)
#%%
# from scipy.signal import hilbert
# #%%
# hb1 = hilbert(coef_tau1[0])
# hb2 = hilbert(coef_tau2[0])


#%%
# n_StimAmp = 1
# n_perStimAmp = 1
# stim_amp = np.array([600])#200*2**np.arange(n_StimAmp)
# #%%
# '''spontanous rate'''
# dt = 1/10000;
# end = int(20/dt); start = int(5/dt)
# spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/15/data.a1.param.Ne
# spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/15/data.a2.param.Ne

# data_anly.spon_rate1 = spon_rate1
# data_anly.spon_rate2 = spon_rate2
# # '''adapt rate'''
# # start = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0] - 2000)/1000/dt))
# # end = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0])/1000/dt))
# # adapt_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/2/data.a1.param.Ne
# # data_anly.adapt_rate1 = adapt_rate1
# # adapt_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/2/data.a2.param.Ne
# # data_anly.adapt_rate2 = adapt_rate2

# title += '_1hz%.2f_2hz%.2f'%(spon_rate1,spon_rate2)

#%%
'''spon'''
#first_stim = 0; last_stim = 0
start_time = 5000 
end_time = 7000

for tr in [0,1]:
    data_tr.a1.ge.i = data.a1.ge[tr]['i']
    data_tr.a1.ge.t = data.a1.ge[tr]['t']
    data_tr.a2.ge.i = data.a2.ge[tr]['i']
    data_tr.a2.ge.t = data.a2.ge[tr]['t']
        
    data_tr.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data_tr.a1.ge.get_centre_mass()
    data_tr.a1.ge.overlap_centreandspike()
    
    data_tr.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data_tr.a2.ge.get_centre_mass()
    data_tr.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data_tr.a1.ge.spk_rate.spk_rate.shape[2]
    
    # stim_on_off = data.a1.param.stim1.stim_on-start_time
    # stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    # #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    # stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    # adpt = None
    # #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    # #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    ani = fra.show_pattern(spkrate1=data_tr.a1.ge.spk_rate.spk_rate, spkrate2=data_tr.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=None, adpt=None)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_spon_tr%d_%d'%(tr,loop_num)+'.mp4'
    
    #ani.save(moviefile)
    del ani

#%%
data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)
#data_anly.save(data_anly.class2dict(), datapath+'data_anly_fano%d.file'%loop_num)
#data_anly.save(data_anly.class2dict(), datapath+'data_anly_fano5hz%d.file'%loop_num)
