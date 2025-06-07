#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:02:51 2021

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
analy_type = 'asso'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly' #'data_anly' data_anly_temp

fftplot = 1; getfano = 1
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1
firing_rate_long = 1

save_img = 1

# if loop_num%4 == 0: save_img = 1
# else: save_img = 0

if loop_num%5 ==0: get_ani = 1
else: get_ani = 0

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()
#%%


if analy_type == 'asso': # fbrg: feedback range
    title = 'hz_2erie%.2f_2irie%.2f_dgk%d'%(data.param.ie_r_e1, data.param.ie_r_i1, \
                                               data.param.new_delta_gk)

#%%
'''spontanous rate'''
dt = 1/10000;
end = int(10/dt); start = int(5/dt)
#spon_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/5/data.a1.param.Ne
spon_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/5/data.a2.param.Ne

#data_anly.spon_rate1 = spon_rate1
data_anly.spon_rate2 = spon_rate2
'''adapt rate'''
# start = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0] - 2000)/1000/dt))
# end = int(round((data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0])/1000/dt))
# adapt_rate1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/2/data.a1.param.Ne
# data_anly.adapt_rate1 = adapt_rate1
# # adapt_rate2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/2/data.a2.param.Ne
# # data_anly.adapt_rate2 = adapt_rate2

#title += '_1hz%.2f_1adphz%.2f_2hz%.2f_2adphz%.2f'%(spon_rate1,adapt_rate1,spon_rate2,adapt_rate2)
title += '_2hz%.2f'%(spon_rate2)


#%%
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a2.param.e_lattice, mua_loca, mua_range, data.a2.param.width)


#%%
start_time = 5e3; end_time = 10e3
window = 5

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = window)
mua = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

coef, freq = fqa.myfft(mua, Fs=1000)
peakF_spon = fqa.find_peakF(coef, freq, 3)
data_anly.coef_spon_a2 = coef
data_anly.freq_spon_a2 = freq
data_anly.peakF_spon_a2 =  peakF_spon

if fftplot and save_img:
    fig, ax = fqa.plot_fft(freq, coef, freq_max1=20, freq_max2 = 200, fig=None, ax=None, label='spon_a2')
    
    title1 = title + '_pf%.2f_spon_a2'%(peakF_spon)
    savetitle = title1.replace('\n','')
    fig.suptitle(title1)
    fftfile = savetitle+'_fft_%d'%(loop_num)+'.png'
    fig.savefig(fftfile)
    plt.close()
    
start_time = 10e3; end_time = 30e3

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = window)
mua = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

coef, freq = fqa.myfft(mua, Fs=1000)
peakF_spon = fqa.find_peakF(coef, freq, 3)
data_anly.coef_adapt_a2 = coef
data_anly.freq_adapt_a2 = freq
data_anly.peakF_adapt_a2 =  peakF_spon

if fftplot and save_img:
    fig, ax = fqa.plot_fft(freq, coef, freq_max1=20, freq_max2 = 200, fig=None, ax=None, label='adapt_a2')
    
    title1 = title + '_pf%.2f_adpt_a2'%(peakF_spon)
    savetitle = title1.replace('\n','')
    fig.suptitle(title1)
    fftfile = savetitle+'_fft_%d'%(loop_num)+'.png'
    fig.savefig(fftfile)
    plt.close()

#%%
'''firing rate long temp'''
# mua_range_1 = 5
# mua_loca_1 = [0,0]
# mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca_1, mua_range_1, data.a1.param.width)

mua_range_2 = 5
mua_loca_2 = [0,0]
mua_neuron_2 = cn.findnearbyneuron.findnearbyneuron(data.a2.param.e_lattice, mua_loca_2, mua_range_2, data.a2.param.width)
window = 10

start_time = 8e3; end_time = 14e3
window = 5

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = window)
mua = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1)[mua_neuron]
mua = mua.mean(0)/(window/1000)

fig, ax = plt.subplots(1,1,figsize=[8,6])
ax.plot(mua)

title1 = title 
savetitle = title1.replace('\n','')
fig.suptitle(title1)
ratefile = savetitle+'_rate_%d'%(loop_num)+'.png'
fig.savefig(ratefile)
plt.close()
#%%
if get_ani:
    '''spon'''
    first_stim = 0; last_stim = 0
    start_time = 8000
    end_time = 12000
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a2.ge.spk_rate.spk_rate.shape[2]
    
    # stim_on_off = data.a1.param.stim1.stim_on-start_time
    # stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    # #stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    # stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    # adpt = None
    # #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    adpt = [[[[31.5,31.5]], [[[2000, data.a2.ge.spk_rate.spk_rate.shape[-1]]]], [[7]]]]
    ani = fra.show_pattern(spkrate1=data.a2.ge.spk_rate.spk_rate, spkrate2=None, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=None, adpt=adpt)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_adpt_%d'%loop_num+'.mp4'
    
    ani.save(moviefile)
    del ani

#%%
data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)

