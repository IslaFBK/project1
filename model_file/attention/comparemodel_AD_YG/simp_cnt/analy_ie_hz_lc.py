#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:24:47 2020

@author: shni2598
"""

import matplotlib as mpl
#mpl.use('Agg')
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
import shutil
from scipy.optimize import curve_fit
import scipy.io as sio
from scipy import signal
#%%
#datapath = '/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/chg_adpt_netsize/'
# import glob
# #path = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ie/'
# path = ''
# FilenamesList = glob.glob(path+'*.mp4')
# path = './good/'
# FilenamesList += glob.glob(path+'*.mp4')
# indList = [None]*len(FilenamesList)
# for i in range(len(FilenamesList)):
#     indList[i] = FilenamesList[i].split('.')[-2].split('_')[-1]
    
# if sys.argv[1] in indList:
#     print('True')
#     sys.exit("Exit, file already exists!")

#%%
analy_type = 'ie_hz'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/ie_hz_5ms/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/gl/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/test2/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/dgk10/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/eeie/ie_hz_nii260/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/ie_hz_dpi1819/revised/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/ie_hz_dgk16tau60/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/ie_hz_dgk16tau60/low_hz_part/'

#%%
data_an = mydata.mydata()
#rate = np.zeros([30,38])
#rate = np.zeros([20,80])
rate = np.zeros([20,98])

#ie = np.arange(2.5,4.0,0.04) *6.5/5.8
#ie = np.arange(2.30, 3.70, 0.04) *6.5/5.8
#ie = np.arange(2.760 -0.7 , 2.760 + 0.7 , 0.04) *6.5/5.8
ie_ratio_part1 = np.arange(2.0, 2.7, 0.04)
ie_ratio_part2 = np.arange(2.70  , 3.5  , 0.01)
ie = np.concatenate((ie_ratio_part1, ie_ratio_part2)) * 6.5/5.8

#ie = np.arange(2.70  , 3.5  , 0.01) * 6.5/5.8
file_id = 0
for ii in range(20):
    for ie_ind in range(98):
        try:
            data_an.load(datapath+'data_anly%d.file'%file_id)
        except FileNotFoundError:
            print(ii, ie_ind)
            file_id += 1
            continue
        rate[ii, ie_ind] = data_an.spon_rate
        file_id += 1
#%%
rate_m = rate[:20].mean(0)
#%%
sio.savemat('rate_m_dgk16tau60.mat', {'rate_m': rate_m,'ie':ie})
#%%
def power_law1(x, a, b, c):
    return a*np.power(x, b) + c
def power_law2(x, a, b):
    return a*np.power(x, b) 
#%%
index1 = np.s_[3:18]; index2 = np.s_[24:61]; index3 = np.s_[63:]; 

pars1, cov1 = curve_fit(f=power_law1, xdata=ie[index1], ydata=rate_m[index1])  #,p0=[-130.9, 0.5404, 260.4 ])
pars2, cov2 = curve_fit(f=power_law1, xdata=ie[index2], ydata=rate_m[index2],p0=[1.239e9, -17.18, 1.968])
pars3, cov3 = curve_fit(f=power_law1, xdata=ie[index3], ydata=rate_m[index3],p0=[8690, -6.901, 0.9625])

#%%
rate1 = power_law1(ie[index1], *pars1)

rate2 = power_law1(ie[index2], *pars2)
rate3 = power_law1(ie[index3], *pars3)

#%%
plt.figure(figsize=[8,5])
plt.plot(ie[:],rate_m[:],'*')
plt.plot(ie[index1], rate1, label='y=a*x^b+c, a={:.4e},b={:.3f}, c={:.3f}'.format(pars1[0],pars1[1],pars1[2]))
plt.plot(ie[index2], rate2, label='y=a*x^b+c, a={:.4e},b={:.3f}, c={:.3f}'.format(pars2[0],pars2[1],pars2[2]))
plt.plot(ie[index3], rate3, label='y=a*x^b+c, a={:.4e},b={:.3f}, c={:.3f}'.format(pars3[0],pars3[1],pars3[2]))
# plt.xlabel('ie')
# plt.ylabel('rate Hz')
plt.legend()
plt.yscale('log')    
#plt.xscale('log')    
plt.ylabel('Hz')
plt.xlabel('ie_ratio')
#plt.title('ie,rate, t_ref:5ms,\nsecond part power law is not well fitted')
#plt.title('ie,rate, t_ref:5ms, gl_inhibitory_neuron_25*nS,\nsecond part power law is not well fitted')
#plt.title('ie,rate, t_ref:5ms, dgk:12nS, \nsecond part power law is not well fitted')
plt.title('''/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/ie_hz_dgk16tau60/
             t_ref:5ms, dgk:16nS, tau_k:60ms, adapt_range:6''')#,\nsecond part power law is not well fitted''')
#plt.title('ee1.20_ei1.30_ie*_ii1.03_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp')
#plt.title('ee1.20_ei1.27_ie*_ii1.00_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/t5_hz_ie')
#%%
plt.figure(figsize=[8,5])
plt.plot(ie[:],rate_m[:],'*')
plt.yscale('log')    

#%%
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/param_try1_5ms/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/param_try1/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/short_couplingrange/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/try_eiii/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/try_pi/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/ie_hz_dpi1819/revised/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/dgk16tau60/'
#%%
data_spec = mydata.mydata()
#coef_spon = []
#coef_adapt = []
#rate = np.zeros([30,38])
#ie = np.arange(2.5,4.0,0.04) *6.5/5.8
del coef_spon, coef_adapt
file_id = 0
for file_id in range(40,60):
    data_spec.load(datapath+'data_anly%d.file'%file_id)
    if 'coef_spon' not in locals():
        coef_spon = np.zeros([20, len(data_spec.coef_spon)],dtype=np.complex128)
    if 'coef_adapt' not in locals():
        coef_adapt = np.zeros([20, len(data_spec.coef_adapt)],dtype=np.complex128)
    coef_spon[file_id-40] = data_spec.coef_spon
    coef_adapt[file_id-40] = data_spec.coef_adapt
    # coef_spon.append(data_spec.coef_spon)
    # coef_adapt.append(data_spec.coef_adapt)
#%%
data_spec = mydata.mydata()
#coef_spon = []
#coef_adapt = []
#rate = np.zeros([30,38])
#ie = np.arange(2.5,4.0,0.04) *6.5/5.8
del coef_spon, coef_adapt
loop_ind = 0
for file_id in range(17,720,36):
    data_spec.load(datapath+'data_anly%d.file'%file_id)
    if 'coef_spon' not in locals():
        coef_spon = np.zeros([20, len(data_spec.coef_spon)],dtype=np.complex128)
    if 'coef_adapt' not in locals():
        coef_adapt = np.zeros([20, len(data_spec.coef_adapt)],dtype=np.complex128)
    coef_spon[loop_ind] = data_spec.coef_spon
    coef_adapt[loop_ind] = data_spec.coef_adapt
    loop_ind += 1
    # coef_spon.append(data_spec.coef_spon)
    # coef_adapt.append(data_spec.coef_adapt)
        
#%%
coef_spon_m = np.zeros(len(coef_spon[0]))#,dtype=np.complex128)
coef_adapt_m = np.zeros(len(coef_adapt[0]))#,dtype=np.complex128)

for i in range(len(coef_spon)):
    coef_spon_m += np.abs(coef_spon[i]) #coef_spon[i]#
    coef_adapt_m += np.abs(coef_adapt[i]) #coef_adapt[i]

coef_spon_m /= 30
coef_adapt_m /= 30
#%%
coef_spon_m = np.abs(coef_spon).mean(0)
coef_adapt_m = np.abs(coef_adapt).mean(0)
#%%
# coef_spon_m = coef_spon.mean(0)
# coef_adapt_m = coef_adapt.mean(0)

#%%
fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000

freq_max = 100
freq = data_spec.freq_spon
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], np.abs(coef_spon_m[1:ind_len]),label='spon')
ax2.loglog(freq[1:ind_len], np.abs(coef_spon_m[1:ind_len]), label='spon_log')
#plt.legend()
#ax1.set_title('spon')

freq_max = 100
freq = data_spec.freq_adapt
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
ax1t=ax1.twinx()
im2 = ax1t.plot(freq[1:ind_len], np.abs(coef_adapt_m[1:ind_len]),'tab:orange',label='adapt')
ax1t.plot([3,3],[0,np.max(np.abs(coef_adapt_m[1:ind_len]))],'r--')
ax1t.plot([8,8],[0,np.max(np.abs(coef_adapt_m[1:ind_len]))],'r--')
ax2.loglog(freq[1:ind_len], np.abs(coef_adapt_m[1:ind_len]),label='adapt_log')
ax2.loglog([3,3],[0,np.max(np.abs(coef_adapt_m[1:ind_len]))],'r--')
ax2.loglog([8,8],[0,np.max(np.abs(coef_adapt_m[1:ind_len]))],'r--')

lms = im1 + im2
labs = [l.get_label() for l in lms]
ax1.legend(lms, labs)#, loc=0)
#ax1.legend()
#ax1t.legend()
ax2.legend()
#ie_ratio = np.linspace(3.07,3.13,7)[4]
# plt.suptitle('''/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/try_eiii/
#              t_ref:5ms, ie_ratio_e:%.3f , power spectrum'''%2.850)
plt.suptitle('''/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/dgk16tau60/
             t_ref:5ms, dgk:16nS, tau:60ms, ie_ratio_e:%.3f , power spectrum'''%2.770)

#%%
fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000

freq_max = 20
freq = data_spec.freq_spon
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], np.abs(coef_spon[0][1:ind_len]),label='spon')
ax2.loglog(freq[1:ind_len], np.abs(coef_spon[0][1:ind_len]),label='spon_log')
#plt.legend()
#ax1.set_title('spon')

freq_max = 20
freq = data_spec.freq_adapt
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
ax1t=ax1.twinx()
im2 = ax1t.plot(freq[1:ind_len], np.abs(coef_adapt[0][1:ind_len]),'tab:orange',label='adapt')
ax1t.plot([3,3],[0,np.max(np.abs(coef_adapt[0][1:ind_len]))],'r--')
ax1t.plot([8,8],[0,np.max(np.abs(coef_adapt[0][1:ind_len]))],'r--')
ax2.loglog(freq[1:ind_len], np.abs(coef_adapt[0][1:ind_len]),label='adapt_log')
ax2.loglog([3,3],[0,np.max(np.abs(coef_adapt[0][1:ind_len]))],'r--')
ax2.loglog([8,8],[0,np.max(np.abs(coef_adapt[0][1:ind_len]))],'r--')

lms = im1 + im2
labs = [l.get_label() for l in lms]
ax1.legend(lms, labs)#, loc=0)
#ax1.legend()
#ax1t.legend()
ax2.legend()
ie_ratio = np.linspace(3.07,3.13,7)[3]
plt.suptitle('t_ref:4ms, ie_ratio_e:%.3f , power spectrum'%ie_ratio)



    # rate[ii, ie_ind] = data_an.spon_rate
    # file_id += 1

#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/ie_hz_5ms/'
loop_num = 813
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
data.a1.ge.get_MSD(start_time=4000, end_time=20000, n_neuron=data.a1.param.Ne, window = 10, jump_interval=np.arange(5,500,10), fit_stableDist='Matlab')
#%%
plt.figure();
plt.loglog(np.arange(5,500,10), data.a1.ge.MSD.MSD,'o')
#%%
datapath = '/run/user/719/gvfs/smb-share:server=shared.sydney.edu.au,share=research-data/PRJ-THAP/shencong/attention/tmp/dp_num_ee/'
loop_num = 550
data2 = mydata.mydata()
data2.load(datapath+'data%d.file'%loop_num)
#%%
start_time = 18e3; end_time = 21e3
data2.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)
#data.a1.ge.get_centre_mass(detect_pattern=True)
#pattern_size2 = data.a1.ge.centre_mass.pattern_size.copy()
#pattern_size2[np.invert(data.a1.ge.centre_mass.pattern)] = 0

frames = data2.a1.ge.spk_rate.spk_rate.shape[2]
# ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
#                                         show_pattern_size=True, pattern_ctr=data.a1.ge.centre_mass.centre_ind, \
#                                             pattern_size=pattern_size2)
ani = firing_rate_analysis.show_pattern(data2.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, show_pattern_size=False)

#%%
data_yg = mydata.mydata(datasave)

#%%
start_time = 0e3; end_time = 1e3
data_yg.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)
#data.a1.ge.get_centre_mass(detect_pattern=True)
#pattern_size2 = data.a1.ge.centre_mass.pattern_size.copy()
#pattern_size2[np.invert(data.a1.ge.centre_mass.pattern)] = 0

frames = data_yg.a1.ge.spk_rate.spk_rate.shape[2]
# ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, anititle=title,\
#                                         show_pattern_size=True, pattern_ctr=data.a1.ge.centre_mass.centre_ind, \
#                                             pattern_size=pattern_size2)
ani = firing_rate_analysis.show_pattern(data_yg.a1.ge.spk_rate.spk_rate, None, frames = frames, start_time = start_time, \
                                        interval_movie=20, anititle=title, show_pattern_size=False)

#%%
file_id = 15
data_an = mydata.mydata()

data_an.load(datapath+'data%d.file'%file_id)
start_time = 4e3; end_time = 20e3
data_an.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_an.a1.param.Ne, window = 10)
#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ni, window = 10)
data_an.a1.ge.get_centre_mass(detect_pattern=False)
#%%
out = np.correlate(data_an.a1.ge.centre_mass.centre[:,0], data_an.a1.ge.centre_mass.centre[:,0],'full')
#%%
plt.figure()
plt.plot(out[int(len(out)/2):])
#%%
plt.figure()
plt.plot(data_an.a1.ge.centre_mass.centre[:,0])
#%%
ctr = np.unwrap(data_an.a1.ge.centre_mass.centre[:,0],31.5)
#%%
plt.figure()
plt.plot(ctr)
#%%
fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000
#data_fft = ctr#data_an.a1.ge.centre_mass.centre[:,0]
data_fft = data_an.a1.ge.centre_mass.centre[:,0]

coef, freq = fa.myfft(data_fft, fs)
#data_anly.coef_spon = coef
#data_anly.freq_spon = freq

#peakF_spon = find_peakF(coef, freq, 3)

freq_max = 20
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon')
ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon_log')
#%%
start_time = 0; end_time = 32e3
data_an.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_an.a1.param.Ne, window = 10)
mua = data_an.a1.ge.spk_rate.spk_rate.reshape(data_an.a1.param.Ne,-1)[data_an.a1.param.chg_adapt_neuron]
mua = mua.mean(0)/0.01

fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000
data_an_fft = mua[4000:20000]
coef, freq = fa.myfft(data_an_fft, fs)
#data_an_anly.coef_spon = coef
#data_an_anly.freq_spon = freq

#peakF_spon = find_peakF(coef, freq, 3)

freq_max = 20
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_an_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon')
ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='spon_log')

#ax1.set_title('spon')

data_an_fft = mua[22000:32000]
coef, freq = fa.myfft(data_an_fft, fs)
#data_an_anly.coef_adapt = coef
#data_an_anly.freq_adapt = freq

#peakF_adapt = find_peakF(coef, freq, 1)

freq_max = 20
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_an_fft)/2)) + 1
ax1t=ax1.twinx()
im2 = ax1t.plot(freq[1:ind_len], np.abs(coef[1:ind_len]),'tab:orange',label='adapt')
ax1t.plot([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax1t.plot([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='adapt_log')
ax2.loglog([3,3],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
ax2.loglog([8,8],[0,np.max(np.abs(coef[1:ind_len]))],'r--')
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/YG/'
loop_num = 7
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
dt = 1/10000;
end = int(14/dt); start = int(4/dt)
np.sum((data.a1.gi.t < end) & (data.a1.gi.t > start))/10/data.a1.param.Ni
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/cfc_dgk16tau60/'
#%%
del MI_raw_mat
del MI_surr_mat
for loop_num in range(40,60):
    data_cfc = mydata.mydata()
    data_cfc.load(datapath+'data_anly%d.file'%loop_num)
    if 'MI_raw_mat' not in locals():
        MI_raw_mat = np.zeros(data_cfc.cfc.MI_raw_mat.shape)
    if 'MI_surr_mat' not in locals():
        MI_surr_mat = np.zeros(data_cfc.cfc.MI_surr_mat.shape)
    MI_raw_mat += data_cfc.cfc.MI_raw_mat
    MI_surr_mat += data_cfc.cfc.MI_surr_mat
    # MI_raw_mat[1] += data_cfc.cfc.MI_raw_mat[1]
    # MI_surr_mat[1] += data_cfc.cfc.MI_surr_mat[1]
    
MI_raw_mat /= 20
MI_surr_mat /= 20
#%%    
phaseBandWid = 0.5#0.49 ;
ampBandWid = 5 ;
phaseBand = np.arange(1,14.1,0.5)
ampBand = np.arange(20,101,5) 

mua_loca = [[0,0],[31.5,31.5]]
#%%

for i in range(len(MI_raw_mat)):
    fig, [ax1,ax2] = plt.subplots(2,1, figsize=[7,9])
    #x_range = np.arange(phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2+1)
    #y_range = np.arange(ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2+1)
    
    #im = ax1.imshow(np.flip(MI_raw_mat[i].T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
    imc = ax1.contourf(phaseBand, ampBand, MI_raw_mat[i].T, 15)#, aspect='auto')
    imcc = ax1.contour(phaseBand, ampBand, MI_raw_mat[i].T, 15, colors='k', linewidths=0.3)#, aspect='auto')
    
    imc2 = ax2.contourf(phaseBand, ampBand, MI_surr_mat[i].T, 15)#, aspect='auto')
    imcc2 = ax2.contour(phaseBand, ampBand, MI_surr_mat[i].T, 15, colors='k', linewidths=0.3)#, aspect='auto')
    
    #imc2 = ax1.contour(phaseBand, ampBand, MI_raw.T, 15)#, aspect='auto')
    
    #imc = ax1.contourf(MI_raw.T, 12, extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#, aspect='auto')
    #imc = ax1.contourf(MI_raw.T, 12, origin='lower', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])#, aspect='auto')
    
    #imi = ax2.imshow(np.flip(MI_raw_mat.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
    plt.colorbar(imc, ax=ax1)
    plt.colorbar(imc2, ax=ax2)
    ax1.set_xlabel('phase frequency (Hz)')
    ax1.set_ylabel('Amplitude frequency (Hz)')
    ax1.set_title('raw')
    ax2.set_xlabel('phase frequency (Hz)')
    ax2.set_ylabel('Amplitude frequency (Hz)')
    ax2.set_title('minus surr')
    plt.suptitle('erie: %.3f, location: %s'%(2.770, mua_loca[i]))
    
    #savetitle = title.replace('\n','')
    
    # fig.suptitle(title)
    # data_anly.peakF_spon = peakF_spon
    # data_anly.peakF_adapt = peakF_adapt
    
    # peakF_good = peakF_spon > 3# and peakF_adapt > 3
    
    #cfcfile = savetitle+'_cfc_%d_%d'%(i,loop_num)+'.png'
    #fig.savefig(cfcfile)
    #plt.close()
#%%
'''trajactory'''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/dgk16tau60/'
#%%
del ac_xtjc_all
#%%
data_tjc = mydata.mydata()
start_loop = 20
for loop_num in range(start_loop,start_loop+20):
    
    data_tjc.load(datapath+'data%d.file'%loop_num)
    
    start_time = 4e3; end_time = 20e3
    data_tjc.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_tjc.a1.param.Ne, window = 10)
    data_tjc.a1.ge.get_centre_mass()
    # mua = data_an.a1.ge.spk_rate.spk_rate.reshape(data_an.a1.param.Ne,-1)[data_tjc.a1.param.chg_adapt_neuron]
    # mua = mua.mean(0)/0.01
    #
    xtjc = data_tjc.a1.ge.centre_mass.centre[:,0]
    
    xtjc -= xtjc.mean()
    ac_xtjc = np.correlate(xtjc, xtjc, 'full')
    ac_xtjc = ac_xtjc[len(xtjc)-1:]
    ac_xtjc /= ac_xtjc[0]
    if 'ac_xtjc_all' not in locals():
        ac_xtjc_all = np.zeros([20,*ac_xtjc.shape])
    ac_xtjc_all[loop_num-start_loop] = ac_xtjc
    
#%%
plt.figure()
plt.plot(ac_xtjc_all.mean(0))
plt.fill_between(np.arange(ac_xtjc_all.shape[1]),ac_xtjc_all.mean(0)-ac_xtjc_all.std(0),ac_xtjc_all.mean(0)+ac_xtjc_all.std(0),color='gray',alpha=0.2)
plt.plot([250,250],[0,1],'r--',label='250ms')
plt.xlabel('ms')
plt.title('autocorrelation of trajectory, ie_ratio: %.3f'%(2.760))
plt.legend()
#%%
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/dgk16tau60/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ieii5tauk/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/power_spectrum/dgk16tau60/'
#%%
loop_num = 1091
data_wvt = mydata.mydata()
data_wvt.load(datapath+'data%d.file'%loop_num)
#%%
start_time = 0; end_time = 32e3
data_wvt.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_wvt.a1.param.Ne, window = 10)
mua = data_wvt.a1.ge.spk_rate.spk_rate.reshape(data_wvt.a1.param.Ne,-1)[data_wvt.a1.param.chg_adapt_neuron]
mua = mua.mean(0)/0.01
#%%
scale_wvt = 2**np.arange(2,12.001,1/10)
wvt_name = 'cmor1.5-1'

coef_wvt, freq_wvt = fa.mycwt(mua, scale_wvt, wvt_name, 1e-3, method = 'fft')
#%%
#extent=[-0.5, coef_wvt.shape[1]-0.5, freq_wvt[-1]/fix_bound, freq_wvt[0]*fix_bound],
fix_bound = 2**(0.5*1/10)
fig = plt.figure()
ax = plt.subplot(111,label='cwt')
ax.imshow(np.abs(coef_wvt), aspect='auto')
ax.yaxis.set_visible(False)
#ax.get_position()
ax2 = fig.add_axes(ax.get_position(),frame_on=False)
ax2.set_ylim([freq_wvt[-1]/fix_bound,freq_wvt[0]*fix_bound])
ax2.xaxis.set_visible(False)

# ax2 = fig.add_subplot(111,label='yaxis')
# #ax2.yaxis.set_ticks(freq_wvt)
# ax2.set_ylim([freq_wvt[-1]/(2**(1/10)),freq_wvt[0]*(2**(1/10))])
# ax2.xaxis.set_visible(False)

plt.yscale('log',base=10)
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/cfc_dgk16tau60/'
#%%
loop_num = 39
data_wvt = mydata.mydata()
data_wvt.load(datapath+'data%d.file'%loop_num)
#%%
start_time = 0; end_time = 26e3
data_wvt.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_wvt.a1.param.Ne, window = 10)
mua = data_wvt.a1.ge.spk_rate.spk_rate.reshape(data_wvt.a1.param.Ne,-1)[data_wvt.a1.param.chg_adapt_neuron]
mua = mua.mean(0)/0.01
#%%
sio.savemat('rate_m_dgk16tau60_sti2.mat', {'mua': mua})#,'ie':ie})

#%%
scale_wvt = 2**np.arange(2,12.001,1/10)
wvt_name = 'cmor1.5-1'

#coef_wvt, freq_wvt = fa.mycwt(mua, scale_wvt, wvt_name, 1e-3, method = 'fft')
coef_wvt, freq_wvt = fa.mycwt(mua, wvt_name, 1e-3, method = 'fft')

#%%
#extent=[-0.5, coef_wvt.shape[1]-0.5, freq_wvt[-1]/fix_bound, freq_wvt[0]*fix_bound],
fix_bound = 2**(0.5*1/10)
fig = plt.figure()
ax = plt.subplot(111,label='cwt')
imcwt = ax.imshow(np.abs(coef_wvt), aspect='auto')
ax.yaxis.set_visible(False)
plt.colorbar(imcwt, ax=ax)

#ax.get_position()
ax2 = fig.add_axes(ax.get_position(),frame_on=False)
ax2.set_ylim([freq_wvt[-1]/fix_bound,freq_wvt[0]*fix_bound])
ax2.xaxis.set_visible(False)

# ax2 = fig.add_subplot(111,label='yaxis')
# #ax2.yaxis.set_ticks(freq_wvt)
# ax2.set_ylim([freq_wvt[-1]/(2**(1/10)),freq_wvt[0]*(2**(1/10))])
# ax2.xaxis.set_visible(False)

plt.yscale('log',base=10)
#%%
fig2 = fa.plot_cwt(coef_wvt, freq_wvt, base=2)
#%%
Wn = np.array([12.5,30])/(1000/2)#subBand[bandNum]/(Fs/2)
filterOrder = 8
sos = signal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')
mua_gamma = signal.sosfiltfilt(sos, mua, axis=0)

Wn = np.array([3,8])/(1000/2)#subBand[bandNum]/(Fs/2)
filterOrder = 8
sos = signal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')
mua_theta = signal.sosfiltfilt(sos, mua, axis=0)
#%%
plt.figure()
# plt.plot(mua_gamma/mua_gamma.max(),label='gamma_60-90')
# plt.plot(mua_theta/mua_theta.max(),label='theta_3-8')
plt.plot(mua_gamma,label='beta_12.5-30')
plt.plot(mua_theta,label='theta_3-8')

#plt.plot([5000,5000],[-1,1],'r--',label='stimuli onset')
plt.plot([5000,5000],[-mua_theta.max(),mua_theta.max()],'r--',label='stimuli onset')

plt.legend()
#%%
plt.figure()
plt.plot(mua_gamma/mua_gamma.max(),label='gamma_60-90')
plt.plot(mua_theta/mua_theta.max(),label='theta_3-8')
plt.plot([5000,5000],[-1,1],'r--',label='stimuli onset')

plt.legend()
#%%

fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000
data_an_fft = mua[4000:19000]
coef, freq = fa.myfft(data_an_fft, fs)
#data_an_anly.coef_spon = coef
#data_an_anly.freq_spon = freq

#peakF_spon = find_peakF(coef, freq, 3)

freq_max = 100
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_an_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label='stim')
im2 = ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='stim')

fig.suptitle('stimuli, fft')
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/YG_stim/'
# datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/cfc_dgk16tau60/'
#%%
loop_num = 7
data_wvt_yg = mydata.mydata()
data_wvt_yg.load(datapath+'data%d.file'%loop_num)
#%%
lattice_ext = cn.coordination.makelattice(int(np.sqrt(data_wvt_yg.a1.param.Ne).round()),data_wvt_yg.a1.param.width,[0,0])

#findcfc = cfc.cfc()
mua_loca = []
mua_loca_n = [0, 0]; mua_loca.append(mua_loca_n)
mua_range = 5#6 * scale_d_p
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(lattice_ext, mua_loca_n, mua_range, data_wvt_yg.a1.param.width)

#%%
start_time = 0; end_time = 20e3
data_wvt_yg.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_wvt_yg.a1.param.Ne, window = 10)
mua = data_wvt_yg.a1.ge.spk_rate.spk_rate.reshape(data_wvt_yg.a1.param.Ne,-1)[mua_neuron_1]
mua = mua.mean(0)/0.01
#%%
sio.savemat('rate_m_dgk16tau60_sti2.mat', {'mua': mua})#,'ie':ie})
#%%
scale_wvt = 2**np.arange(2,12.001,1/10)
wvt_name = 'cmor1.5-1'

#coef_wvt, freq_wvt = fa.mycwt(mua, scale_wvt, wvt_name, 1e-3, method = 'fft')
coef_wvt, freq_wvt = fa.mycwt(mua, wvt_name, 1e-3, method = 'fft')

#%%
#extent=[-0.5, coef_wvt.shape[1]-0.5, freq_wvt[-1]/fix_bound, freq_wvt[0]*fix_bound],
fix_bound = 2**(0.5*1/10)
fig = plt.figure()
ax = plt.subplot(111,label='cwt')
imcwt = ax.imshow(np.abs(coef_wvt), aspect='auto')
ax.yaxis.set_visible(False)
plt.colorbar(imcwt, ax=ax)

#ax.get_position()
ax2 = fig.add_axes(ax.get_position(),frame_on=False)
ax2.set_ylim([freq_wvt[-1]/fix_bound,freq_wvt[0]*fix_bound])
ax2.xaxis.set_visible(False)

# ax2 = fig.add_subplot(111,label='yaxis')
# #ax2.yaxis.set_ticks(freq_wvt)
# ax2.set_ylim([freq_wvt[-1]/(2**(1/10)),freq_wvt[0]*(2**(1/10))])
# ax2.xaxis.set_visible(False)

plt.yscale('log',base=10)
#%%
fig2 = fa.plot_cwt(coef_wvt, freq_wvt, base=2)
#%%
Wn = np.array([30,60])/(1000/2)#subBand[bandNum]/(Fs/2)
filterOrder = 8
sos = signal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')
mua_gamma = signal.sosfiltfilt(sos, mua, axis=0)

Wn = np.array([3,8])/(1000/2)#subBand[bandNum]/(Fs/2)
filterOrder = 8
sos = signal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')
mua_theta = signal.sosfiltfilt(sos, mua, axis=0)
#%%
plt.figure()
# plt.plot(mua_gamma/mua_gamma.max(),label='gamma_60-90')
# plt.plot(mua_theta/mua_theta.max(),label='theta_3-8')
plt.plot(mua_gamma,label='gamma_60-90')
plt.plot(mua_theta,label='theta_3-8')

#plt.plot([5000,5000],[-1,1],'r--',label='stimuli onset')
plt.plot([5000,5000],[-mua_theta.max(),mua_theta.max()],'r--',label='stimuli onset')

plt.legend()
#%%
plt.figure()
plt.plot(mua_gamma/mua_gamma.max(),label='gamma_60-90')
plt.plot(mua_theta/mua_theta.max(),label='theta_3-8')
plt.plot([5000,5000],[-1,1],'r--',label='stimuli onset')

plt.legend()
#%%

fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000
data_an_fft = mua[4000:19000]
coef, freq = fa.myfft(data_an_fft, fs)
#data_an_anly.coef_spon = coef
#data_an_anly.freq_spon = freq

#peakF_spon = find_peakF(coef, freq, 3)

freq_max = 100
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_an_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label='stim')
im2 = ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='stim')

fig.suptitle('stimuli, fft')
#%%
fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000
data_an_fft = mua[4000:19000]
coef, freq = fa.myfft(data_an_fft, fs)
#data_an_anly.coef_spon = coef
#data_an_anly.freq_spon = freq

#peakF_spon = find_peakF(coef, freq, 3)

freq_max = 100
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_an_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label='stim')
im2 = ax2.loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label='stim')

fig.suptitle('stimuli, fft')
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/YG_stim/'
# datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/cfc_dgk16tau60/'
#%%
mua_loca = []
mua_loca_n = [0, 0]; mua_loca.append(mua_loca_n)
mua_range = 5#6 * scale_d_p
mua_neuron_1 = cn.findnearbyneuron.findnearbyneuron(lattice_ext, mua_loca_n, mua_range, data_wvt_yg.a1.param.width)
#%%
timebin = 10

for loop_num in range(5):
    data_wvt_yg = mydata.mydata()
    data_wvt_yg.load(datapath+'data%d.file'%loop_num)
    start_time = 0; end_time = 20e3
    data_wvt_yg.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_wvt_yg.a1.param.Ne, window = timebin)
    mua = data_wvt_yg.a1.ge.spk_rate.spk_rate.reshape(data_wvt_yg.a1.param.Ne,-1)[mua_neuron_1]
    mua = mua.mean(0)/(timebin/1000)
    data_an_fft = mua[4000:19000]
    fs = 1000
    coef, freq = fa.myfft(data_an_fft, fs)
    if 'coef_muti_yg' not in locals():
        coef_muti_yg = np.zeros(coef.shape)#, np.complex128)
    coef_muti_yg += np.abs(coef)

coef_muti_yg /= 5    
#%%
fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000
#data_an_fft = mua[4000:19000]
#coef, freq = fa.myfft(data_an_fft, fs)
#data_an_anly.coef_spon = coef
#data_an_anly.freq_spon = freq

#peakF_spon = find_peakF(coef, freq, 3)

freq_max = 200
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_an_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], coef_muti_yg[1:ind_len],label='stim')
im2 = ax2.loglog(freq[1:ind_len], coef_muti_yg[1:ind_len],label='stim')

fig.suptitle('yifan_gc_param, stimuli, fft, bin:10ms')
#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/ie_hz/cfc_dgk16tau60/'
del coef_muti_gz
timebin = 5
for loop_num in range(20,40):
    data_wvt_gz = mydata.mydata()
    data_wvt_gz.load(datapath+'data%d.file'%loop_num)
    start_time = 0; end_time = 26e3
    data_wvt_gz.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data_wvt_gz.a1.param.Ne, window = timebin)
    mua = data_wvt_gz.a1.ge.spk_rate.spk_rate.reshape(data_wvt_gz.a1.param.Ne,-1)[mua_neuron_1]
    mua = mua.mean(0)/(timebin/1000)
    data_an_fft = mua[5000:26000]
    fs = 1000
    coef, freq = fa.myfft(data_an_fft, fs)
    if 'coef_muti_gz' not in locals():
        coef_muti_gz = np.zeros(coef.shape)#, np.complex128)
    coef_muti_gz += np.abs(coef)

coef_muti_gz /= 20    
#%%
fig, [ax1,ax2] = plt.subplots(1,2,figsize=[9,4])

fs = 1000
#data_an_fft = mua[4000:19000]
#coef, freq = fa.myfft(data_an_fft, fs)
#data_an_anly.coef_spon = coef
#data_an_anly.freq_spon = freq

#peakF_spon = find_peakF(coef, freq, 3)

freq_max = 100
ind_len = freq[freq<freq_max].shape[0] # int(20/(fs/2)*(len(data_an_fft)/2)) + 1
im1 = ax1.plot(freq[1:ind_len], coef_muti_gz[1:ind_len],label='stim')
im2 = ax2.loglog(freq[1:ind_len], coef_muti_gz[1:ind_len],label='stim')

fig.suptitle('stimuli, fft, bin:5ms')
