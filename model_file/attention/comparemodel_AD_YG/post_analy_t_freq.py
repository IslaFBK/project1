#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:40:30 2020

@author: shni2598
"""

import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import connection as cn
import firing_rate_analysis
import frequency_analysis
import pickle
import sys
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import stats
from scipy import signal as spsignal
import levy
#%%
loop_num = int(sys.argv[1])

datapath = ''
data6 = mydata.mydata()
data6.load(datapath+'data%d.file'%loop_num)
#%%
dt = data6.dt/1000
Fs = 1/dt


data6.a1.ge.get_sparse_spk_matrix(shape=[3969, int(np.round(data6.a1.param.simutime/0.1))])
#%%
lattice_ext = cn.coordination.makelattice(63,62,[0,0])
mua_loca = [0, 0]
mua_range = 5
mua_neuron = cn.findnearbyneuron.findnearbyneuron(lattice_ext, mua_loca, mua_range, 62)

mua = data6.a1.ge.spk_matrix[mua_neuron,:].A.sum(0)
mua_ker = np.ones(20);
mua_rate = np.convolve(mua, mua_ker, mode='valid')
#%%
#plt.figure()
#plt.plot(np.arange(int(np.round(data6.a1.param.simutime/0.1)))*0.1, mua)
#%%
startstep = int(np.round(8/dt)); endstep = int(np.round(21/dt))
fig, [ax1,ax2] = plt.subplots(2,1,figsize=[15,6])
ax1.plot(np.arange(len(mua_rate[startstep:endstep]))*0.1, mua_rate[startstep:endstep])

#%%
'''
Theta_MUA
'''
Fs = 10000
low_f = 3; high_f = 7;
Wn = np.array([low_f, high_f])/(Fs/2)
sos = spsignal.butter(4, Wn, 'bandpass', output='sos')
#%%
mua_rate_low = spsignal.sosfilt(sos, mua_rate[startstep:endstep])
#plt.figure()
#plt.plot(0.1*np.arange(len(mua_rate_low)), mua_rate_low)
#%%
'''
Gamma_MUA
'''
low_f = 30; high_f = 80;
Wn = np.array([low_f, high_f])/(Fs/2)
sos = spsignal.butter(4, Wn, 'bandpass', output='sos')
#%%
mua_rate_high = spsignal.sosfilt(sos,mua_rate[startstep:endstep])
#plt.figure()
#plt.plot(0.1*np.arange(len(mua_rate_high)), mua_rate_high)
#%%
'''
together
'''
#plt.figure()
ax2.plot(0.1*np.arange(len(mua_rate_low)),mua_rate_low/(np.max(mua_rate_low) - np.min(mua_rate_low)))
ax2.plot(0.1*np.arange(len(mua_rate_high)),mua_rate_high/(np.max(mua_rate_high) - np.min(mua_rate_high)))

#scale_ee_ei_ii = data6.a1.param.mean_J_ee/(4*10**-3)
#fig.suptitle('MUA_other:%.4f_ie:%.3f'%(scale_ee_ei_ii,data6.a1.param.ie_ratio))
fig.suptitle('MUA_ii:%.4f_ie:%.3f'%(data6.a1.param.w_ii,data6.a1.param.ie_ratio))

#fig.savefig('MUA_%.4f_%.3f_%d.svg'%(data6.a1.param.mean_J_ee,data6.a1.param.ie_ratio,loop_num))
#fig.savefig('MUA_other:%.4f_ie:%.3f_%d.png'%(scale_ee_ei_ii,data6.a1.param.ie_ratio,loop_num))
fig.savefig('MUA_ii:%.4f_ie:%.3f_%d.png'%(data6.a1.param.w_ii,data6.a1.param.ie_ratio,loop_num))

#%%
'''wavelet'''
dt=1/1e4
noctave = 9;
nvoice = 10
minscale = 64
scale = minscale * np.power(2, 1/nvoice*np.arange(0, noctave*nvoice+1))
coef_cwt, freq_cwt = frequency_analysis.mycwt(mua_rate, scale, 'cmor1.5-1.0', dt, method='fft', L1_norm = True)
#%%
#coef_cwt_norm = stats.zscore(coef_cwt, axis = 1)
#%%
fig, ax1 = plt.subplots(1,1,figsize=[8,5])
s_per = 2**(1/(2*nvoice))
cwt_plt = ax1.imshow(np.abs(coef_cwt), aspect='auto', extent=[0,coef_cwt.shape[1]-1, freq_cwt.min()/s_per, freq_cwt.max()*s_per])
plt.colorbar(cwt_plt)

ax1.yaxis.tick_right()
ax1.tick_params(right=False)
ax1.tick_params(labelright=False)

ax2 = fig.add_axes(ax1.get_position(), frameon=False)
#ax2 = ax1.twinx()
#ax2.yaxis.set_label_position('left')
ax2.tick_params(bottom=False,labelbottom=False)
#ax2.yaxis.tick_right()
ax2.set_ylim([freq_cwt.min()/s_per,freq_cwt.max()*s_per])
ax2.set_yscale('log', basey=10)
#fig.suptitle('MUA_wvt_other:%.4f_ie:%.3f'%(scale_ee_ei_ii,data6.a1.param.ie_ratio))
#fig.savefig('MUA_wvt_%.4f_%.3f_%d.svg'%(data6.a1.param.mean_J_ee,data6.a1.param.ie_ratio,loop_num))
#fig.savefig('MUA_wvt_other:%.4f_ie:%.3f_%d.png'%(scale_ee_ei_ii,data6.a1.param.ie_ratio,loop_num))

fig.suptitle('MUA_wvt_ii:%.4f_ie:%.3f'%(data6.a1.param.w_ii,data6.a1.param.ie_ratio))
fig.savefig('MUA_wvt_ii:%.4f_ie:%.3f_%d.png'%(data6.a1.param.w_ii,data6.a1.param.ie_ratio,loop_num))




