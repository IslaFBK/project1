#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:11:09 2020

@author: shni2598
"""

import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis
import frequency_analysis as fa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
#%%
#datapath = '/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/'
datapath = ''
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
'''spon rate'''
data_anly = mydata.mydata()

dt = 1/10000;
end = int(10/dt); start = int(4/dt)
spon_rate = np.sum((data.a1.ge.t < end) & (data.a1.ge.t > start))/6/data.a1.param.Ne
data_anly.spon_rate = spon_rate
data_anly.save(data_anly.class2dict(), datapath+'data_anly%d.file'%loop_num)

'''animation'''

start_time = 4e3; end_time = 6e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, window = 10)
data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = 1024, window = 10)

data.a1.ge.get_MSD(start_time=4000, end_time=10000, window = 15, jump_interval=np.array([15]), fit_stableDist='pylevy')
#title = "ee:{:.3f}, ie:{:.3f}, decay_p_i:{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.decay_p_ie)
#title = "ee:{:.3f}, ie:{:.3f}, dgk:{:.1f}, alpha:{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.delta_gk,data.a1.ge.MSD.stableDist_param[0,0])
title = "cn{:d}_ee{:.3f}_ie{:.3f}_ei{:.2f}_ii{:.2f}_alpha{:.2f}_hz{:.2f}".format(data.a1.param.cn_scale_wire,data.a1.param.scale_ee_1,data.a1.param.ie_ratio,data.a1.param.scale_ei_1,\
                                                                          data.a1.param.scale_ii_1,data.a1.ge.MSD.stableDist_param[0,0],spon_rate)

frames = data.a1.ge.spk_rate.spk_rate.shape[2]
ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, data.a1.gi.spk_rate.spk_rate, frames = frames, start_time = start_time, anititle=title)
#data.a1.param.scale_ie_1

#savetitle = "ee{:.3f}_ie{:.3f}_dpi{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.decay_p_ie)
#savetitle = "ee{:.3f}_ie{:.3f}_dgk{:.1f}_alpha{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.delta_gk,data.a1.ge.MSD.stableDist_param[0,0])
savetitle = title
ani.save(savetitle+'_%d'%loop_num+'.mp4')
del data.a1.ge.spk_rate

#title = "ee:{:.3f}, ie:{:.3f}, decay_p_i:{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.decay_p_ie)
#savetitle = "ee{:.3f}_ie{:.3f}_dpi{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.decay_p_ie)

#title = "ee:{:.3f}, ie:{:.3f}, dgk:{:.1f}, alpha:{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.delta_gk,data.a1.ge.MSD.stableDist_param[0,0])
#savetitle = "ee{:.3f}_ie{:.3f}_dgk{:.1f}_alpha{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.delta_gk,data.a1.ge.MSD.stableDist_param[0,0])

#title = "ee{:.3f}_ie{:.3f}_ei{:.2f}_ii{:.2f}_alpha{:.2f}_hz{:.2f}".format(data.a1.param.scale_ee_1,data.a1.param.scale_ie_1,data.a1.param.scale_ei_1,data.a1.param.scale_ii_1,data.a1.ge.MSD.stableDist_param[0,0],spon_rate)
savetitle = title

#%%

'''rate'''
e_lattice = cn.coordination.makelattice(63,62,[0,0])

chg_adapt_loca = [0, 0]
chg_adapt_range = 6
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(e_lattice, chg_adapt_loca, chg_adapt_range, 62)
#%%
start_time = 8e3; end_time = 18e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, window = 10)
mua = data.a1.ge.spk_rate.spk_rate.reshape(3969,-1)[chg_adapt_neuron]
mua = mua.mean(0)/0.01

fig, ax1 = plt.subplots(1,1)
ax1.plot(np.arange(len(mua))+start_time, mua)
ax1.set_title(title)

fig.savefig(savetitle+'_rate_%d'%loop_num+'.png')

del data.a1.ge.spk_rate


'''fft'''

start_time = 0; end_time = 20e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
mua = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1)[chg_adapt_neuron]
mua = mua.mean(0)/0.01

fig, [ax1,ax2] = plt.subplots(1,2,figsize=[12,5])

fs = 1000
data_fft = mua[4000:10000]
coef, freq = fa.myfft(data_fft, fs)

freq_max = 50
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax1.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')

ax1.set_title('spon')

data_fft = mua[12000:20000]
coef, freq = fa.myfft(data_fft, fs)

freq_max = 50
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
ax2.plot(freq[1:ind_len], np.abs(coef[1:ind_len]))
ax2.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')

ax2.set_title('adapt')

fig.savefig(savetitle+'_fft_%d'%loop_num+'.png')

#%%


''' MSD '''
data.a1.ge.get_MSD(start_time=4000, end_time=10000, window = 15, jump_interval=np.arange(1,1000,5), fit_stableDist='Matlab')

#data.save(data.class2dict(), datapath+'data%d.file'%loop_num)
#%%
fig, ax1 = plt.subplots(1,1)
ax1.loglog(data.a1.ge.MSD.jump_interval, data.a1.ge.MSD.MSD)

ax2 = ax1.twinx()
ax2.set_ylim([0,2.5])
err_up = data.a1.ge.MSD.stableDist_param[:,2,0] - data.a1.ge.MSD.stableDist_param[:,0,0]
err_down = data.a1.ge.MSD.stableDist_param[:,0,0] - data.a1.ge.MSD.stableDist_param[:,1,0]

googfit = np.abs(err_up - err_down)/(err_up + err_down) < 0.01
ax2.errorbar(data.a1.ge.MSD.jump_interval[googfit], \
             data.a1.ge.MSD.stableDist_param[googfit,0,0], \
             yerr=data.a1.ge.MSD.stableDist_param[googfit,2,0] - data.a1.ge.MSD.stableDist_param[googfit,0,0], fmt='ro')
#ax2.errorbar(data.a1.ge.MSD.jump_interval, \
#             data.a1.ge.MSD.stableDist_param[:], yerr=0)

ax1.set_title(title)
fig.savefig(savetitle+'_msd_%d'%loop_num+'.png')

#%%
# data.save(data.class2dict(), datapath+'data%d.file'%loop_num)

