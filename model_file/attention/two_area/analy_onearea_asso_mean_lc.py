#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:38:43 2021

@author: shni2598
"""

import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fra
import frequency_analysis as fqa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
from cfc_analysis import cfc
#%%
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/onearea_4096/verify/asso_electrode1/'
sys_argv = 0#int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)

#%%
data_anly = mydata.mydata()
loop_num = 0
#data_anly_file = 'data_anly_electrode2_'
data_anly_file = 'data_anly'

data_anly.load(datapath+'%s%d.file'%(data_anly_file, loop_num))

freq_spon = data_anly.freq_spon
coef_spon_mean = np.zeros(data_anly.coef_spon.shape)
freq_adapt = data_anly.freq_adapt
coef_adapt_mean = np.zeros(data_anly.coef_adapt.shape)

spon_rate = 0

MI_raw = np.zeros(data_anly.cfc.MI_raw.shape)
MI_surr = np.zeros(data_anly.cfc.MI_surr.shape)

for loop_num in range(10):
    data_anly.load(datapath+'%s%d.file'%(data_anly_file, loop_num))
    coef_spon_mean += np.abs(data_anly.coef_spon)
    coef_adapt_mean += np.abs(data_anly.coef_adapt)
    MI_raw += data_anly.cfc.MI_raw
    MI_surr += data_anly.cfc.MI_surr
    spon_rate += data_anly.spon_rate
#%%
coef_spon_mean /= 10
coef_adapt_mean /= 10
MI_raw /= 10
MI_surr /= 10
spon_rate /= 10
#%%
def plot_fft(freq, coef, freq_max1=20, freq_max2 = 200, fig=None, ax=None, label=''):
    if fig is None:
        fig, ax = plt.subplots(2,2,figsize=[9,9])
    
    #fs = 1000
    #data_fft = mua[:]
    #coef, freq = fqa.myfft(data_fft, fs)
    # data_anly.coef_spon = coef
    # data_anly.freq_spon = freq
    
    peakF = find_peakF(coef, freq, 3)
    
    #freq_max1 = 20
    ind_len = freq[freq<freq_max1].shape[0] # int(20/(fs/2)data_fft*(len(data_fft)/2)) + 1
    ax[0,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_linear')
    ax[0,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_loglog')
    
    #freq_max2 = 150
    ind_len = freq[freq<freq_max2].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
    ax[1,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_linear')
    ax[1,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_loglog')
    
    for i in range(2):
        for j in range(2):
            ax[i,j].legend()
    
    return fig, ax, peakF#, coef, freq
#%%
fig, ax = plt.subplots(2,2,figsize=[9,9])
fig, ax, peakF_spon = plot_fft(freq_spon, coef_spon_mean, freq_max1=20, freq_max2 = 200, fig=fig, ax=ax, label='spon')
fig, ax, peakF_spon = plot_fft(freq_adapt, coef_adapt_mean, freq_max1=20, freq_max2 = 200, fig=fig, ax=ax, label='adapt')
#%%
peakF_spon = freq_spon[1:][np.argmax(coef_spon_mean[1:])]
peakF_adapt = freq_adapt[1:][np.argmax(coef_adapt_mean[1:])]

fig.suptitle('mean spectrum(10 realizations)\npeak_freq_spon:%.3f Hz; peak_freq_adapt:%.3f Hz'%(peakF_spon, peakF_adapt))

#%%
phaseBand = data_anly.cfc.phaseBand
ampBand = data_anly.cfc.ampBand[2:]
fig, [ax1,ax2] = plt.subplots(2,1, figsize=[7,9])
#x_range = np.arange(phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2+1)
#y_range = np.arange(ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2+1)

#im = ax1.imshow(np.flip(MI_raw.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid/2, phaseBand[-1]+phaseBandWid/2, ampBand[0]-ampBandWid/2, ampBand[-1]+ampBandWid/2])
imc = ax1.contourf(phaseBand, ampBand, MI_raw[:,2:].T, 15)#, aspect='auto')
imcc = ax1.contour(phaseBand, ampBand, MI_raw[:,2:].T, 15, colors='k', linewidths=0.3)#, aspect='auto')

imc2 = ax2.contourf(phaseBand, ampBand, MI_surr[:,2:].T, 15)#, aspect='auto')
imcc2 = ax2.contour(phaseBand, ampBand, MI_surr[:,2:].T, 15, colors='k', linewidths=0.3)#, aspect='auto')

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
ax2.set_title('minus-surr')
#plt.suptitle('ee1.20_ei1.27_ie1.2137_ii1.08_dsi4.44_dse5.00\n/headnode1/shni2598/brian2/NeuroNet_brian/\nmodel_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp')

#titlecfc = title + '_cfc'
plt.suptitle(titlecfc)