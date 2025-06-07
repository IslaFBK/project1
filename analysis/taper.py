#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:18:34 2021

@author: shni2598
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, freqz


import matplotlib as mpl
mpl.use('Agg')
from scipy.stats import sem
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
M = 51
fig, axes = plt.subplots(3, 2, figsize=(5, 7))
for ai, alpha in enumerate((1, 3, 5)):
    win_dpss = windows.dpss(M, alpha)
    beta = alpha*np.pi
    win_kaiser = windows.kaiser(M, beta)
    for win, c in ((win_dpss, 'k'), (win_kaiser, 'r')):
        win /= win.sum()
        axes[ai, 0].plot(win, color=c, lw=1.)
        axes[ai, 0].set(xlim=[0, M-1], title=r'$\alpha$ = %s' % alpha,
                        ylabel='Amplitude')
        w, h = freqz(win)
        axes[ai, 1].plot(w, 20 * np.log10(np.abs(h)), color=c, lw=1.)
        axes[ai, 1].set(xlim=[0, np.pi],
                        title=r'$\beta$ = %0.2f' % beta,
                        ylabel='Magnitude (dB)')
for ax in axes.ravel():
    ax.grid(True)
axes[2, 1].legend(['DPSS', 'Kaiser'])
fig.tight_layout()
#plt.show()


#%%
M = 51; alpha = 1.2
win_dpss = windows.dpss(M, alpha, 2)
win_dpss /= win_dpss.sum()
w, h = freqz(win_dpss)

fig, axes = plt.subplots(1, 2, figsize=(5, 3))

axes[0].plot(win_dpss, lw=1.)
axes[1].plot(w, 20 * np.log10(np.abs(h)), lw=1.)
#%%
M = 320; alpha = 4
#win = windows.dpss(M, alpha, 4, sym=False)
win = windows.dpss(M, alpha, 7, sym=True)
#%%
win_dpss = win[0]
win_dpss /= win_dpss.sum()
w, h = freqz(win_dpss)

fig, axes = plt.subplots(1, 2, figsize=(5, 3))

axes[0].plot(win_dpss, lw=1.)
axes[1].plot(w, 20 * np.log10(np.abs(h)), lw=1.)

#%%
data_dir = 'raw_data/'
#save_dir = 'mean_results/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
analy_type = 'state'
#datapath = data_dir
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/test_longstim2/'+data_dir

data = mydata.mydata()
loop_num = 0
data.load(datapath+'data%d.file'%loop_num)

#%%
#stim_onoff = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
stim_onoff = data.a1.param.stim1.stim_on[:3].copy()
stim_onoff[:,0] += 200

dura_persample = 1500
stim_onoff_new_list = []
for dura in stim_onoff:
    stim_onoff_new_list.append(np.arange(dura[0], dura[1]+1, dura_persample)[:-1])

stim_onoff_new_tmp = np.concatenate(stim_onoff_new_list)
stim_onoff_new = np.zeros([stim_onoff_new_tmp.shape[0],2], int)
stim_onoff_new[:,0] = stim_onoff_new_tmp
stim_onoff_new[:,1] = stim_onoff_new_tmp + dura_persample

#%%
stim_onoff = data.a1.param.stim1.stim_on[3:].copy()
stim_onoff[:,0] += 200

dura_persample = 1500
stim_onoff_new_list = []
for dura in stim_onoff:
    stim_onoff_new_list.append(np.arange(dura[0], dura[1]+1, dura_persample)[:-1])

stim_onoff_new_tmp = np.concatenate(stim_onoff_new_list)
stim_onoff_new_att = np.zeros([stim_onoff_new_tmp.shape[0],2], int)
stim_onoff_new_att[:,0] = stim_onoff_new_tmp
stim_onoff_new_att[:,1] = stim_onoff_new_tmp + dura_persample

#%%
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)

#%%

spk_window = 5

simu_time_tot = data.param.simutime
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10])


mua_1 = fra.get_spkcount_sum_sparmat(data.a1.ge.spk_matrix[mua_neuron].copy(), start_time=0, end_time=simu_time_tot,\
                   sample_interval = 1,  window = spk_window, dt = 0.1)
mua_2 = fra.get_spkcount_sum_sparmat(data.a2.ge.spk_matrix[mua_neuron].copy(), start_time=0, end_time=simu_time_tot,\
                   sample_interval = 1,  window = spk_window, dt = 0.1)

#%%
plt.figure()
plt.plot(mua_1[20000:22000])
plt.plot(mua_2[20000:22000])


#%%
t = np.arange(2000)/1000
d1m = np.zeros([10, 2000])
d2m = np.zeros([10, 2000])
for i in range(10):    
    d1m[i] = np.sin(2*np.pi*20*t) + np.random.randn(2000)*0.1
    d2m[i] = np.sin(2*np.pi*20*t-np.pi/2) + np.random.randn(2000)*0.1

#%%
plt.figure()
plt.plot(d)
#%%
'''
nw = T*hW_real = N*hW_digi = N*(hW_real/Fs) 
nw : (or TW) time-half-bandwidth product
T: length of signal (in second)
hW_real: half-bandwith (in Hz)
N: length of signal (number, dimensionless)
hW_digi: half-bandwith (digital freqency, dimensionless)
Fs: sampling frequency (Hz)
'''

hW_real = 10 # hz 3
Fs = 1000 # hz

T = 1000; # ms
hb = 10 # hz

nw = int(round(dura_persample*hW_real/Fs))
ntap = int(np.floor(2*nw - 1))
#%%
win_dpss = windows.dpss(dura_persample, nw, ntap, sym=False)

tap_data_1 = win_dpss*d1
tap_data_2 = win_dpss*d2


#%%
n_data = 10

s12 = np.zeros(d1.shape, dtype=complex)
s1 = np.zeros(d1.shape, dtype=float)
s2 = np.zeros(d1.shape, dtype=float)
#%%
for d1, d2, in zip(d1m, d2m):
    #print(d1.shape,d2.shape)
    tap_data_1 = win_dpss*d1
    tap_data_2 = win_dpss*d2
    
    coef_1 = np.fft.fft(tap_data_1, axis = -1)
    coef_2 = np.fft.fft(tap_data_2, axis = -1)
    
    s12 += np.mean(coef_1 * np.conjugate(coef_2), 0)/n_data
    s1 += np.mean(np.abs(coef_1)**2, 0)/n_data
    s2 += np.mean(np.abs(coef_2)**2, 0)/n_data

#%%
cohe = np.abs(s12/np.sqrt(s1*s2))
phase = np.angle(s12/np.sqrt(s1*s2))
#%%
plt.figure()
plt.plot(Fs*np.arange(2000)/2000, cohe)
plt.plot(Fs*np.arange(2000)/2000, phase)
#%%
plt.figure()

plt.hist(phase[cohe>0.2])

#%%
spec = np.mean(np.abs(coef)**2, 0)
#%%
plt.figure()
plt.plot(spec)
#%%
plt.figure()
plt.plot(np.abs(coef)/coef.shape[0])

plt.figure()
plt.plot(np.angle(coef))

#%%
n_data = stim_onoff_new.shape[0]

if dura_persample%2 == 0: fft_len = int(round(dura_persample/2)) + 1
else: fft_len = int(round((dura_persample+1)/2))
   
s12 = np.zeros(fft_len, dtype=complex)
s1 = np.zeros(fft_len, dtype=float)
s2 = np.zeros(fft_len, dtype=float)



#%%
for dura in stim_onoff_new:
    #print(d1.shape,d2.shape)
    tap_data_1 = win_dpss*mua_1[dura[0]:dura[1]]
    tap_data_2 = win_dpss*mua_2[dura[0]:dura[1]]
    
    coef_1 = np.fft.rfft(tap_data_1, axis = -1)
    coef_2 = np.fft.rfft(tap_data_2, axis = -1)
    
    s12 += np.mean(coef_1 * np.conjugate(coef_2), 0)/n_data
    s1 += np.mean(np.abs(coef_1)**2, 0)/n_data
    s2 += np.mean(np.abs(coef_2)**2, 0)/n_data
#%%
cohe = np.abs(s12/np.sqrt(s1*s2))
phase = np.angle(s12/np.sqrt(s1*s2))
#%%
fig, ax = plt.subplots(2,1)
ax[0].plot(Fs*np.arange(fft_len)/fft_len, cohe)
ax[1].plot(Fs*np.arange(fft_len)/fft_len, phase)
#%%


n_data = stim_onoff_new_att.shape[0]

s12_att = np.zeros(fft_len, dtype=complex)
s1_att = np.zeros(fft_len, dtype=float)
s2_att = np.zeros(fft_len, dtype=float)



#%%
for dura in stim_onoff_new_att:
    #print(d1.shape,d2.shape)
    tap_data_1 = win_dpss*mua_1[dura[0]:dura[1]]
    tap_data_2 = win_dpss*mua_2[dura[0]:dura[1]]
    
    coef_1 = np.fft.rfft(tap_data_1, axis = -1)
    coef_2 = np.fft.rfft(tap_data_2, axis = -1)
    
    s12_att += np.mean(coef_1 * np.conjugate(coef_2), 0)/n_data
    s1_att += np.mean(np.abs(coef_1)**2, 0)/n_data
    s2_att += np.mean(np.abs(coef_2)**2, 0)/n_data
#%%
cohe_att = np.abs(s12_att/np.sqrt(s1_att*s2_att))
phase_att = np.angle(s12_att/np.sqrt(s1_att*s2_att))
#%%
fig, ax = plt.subplots(2,1)
ax[0].plot(Fs*np.arange(fft_len)/2*fft_len, cohe_att)
ax[1].plot(Fs*np.arange(fft_len)/2*fft_len, phase_att)
#%%
fig, ax = plt.subplots(2,1)
ax[0].plot(Fs*np.arange(fft_len)/(2*fft_len), cohe)
ax[0].plot(Fs*np.arange(fft_len)/(2*fft_len), cohe_att)

ax[1].plot(Fs*np.arange(fft_len)/(2*fft_len), phase)
ax[1].plot(Fs*np.arange(fft_len)/(2*fft_len), phase_att)
#%%
cc1 = np.fft.fft(np.arange(10))
cc2 = np.fft.rfft(np.arange(10))

fig, ax = plt.subplots(2,1)
ax[0].plot(np.abs(cc1))
ax[1].plot(np.abs(cc2))
#%%
fig, ax = plt.subplots(2,1)
ax[0].plot(np.angle(cc1))
ax[1].plot(np.angle(cc2))
#%%
#%%
class taper_coherence_multisignal:
    
    def __init__(self):
        '''
        nw = T*hW_real = N*hW_digi = N*(hW_real/Fs) 
        nw : (or TW) time-half-bandwidth product
        T: length of signal (in second)
        hW_real: half-bandwith (in Hz)
        N: length of signal (number, dimensionless)
        hW_digi: half-bandwith (digital freqency, dimensionless)
        Fs: sampling frequency (Hz)
        '''
        
        self.hW_real = 10 # hz 3
        self.Fs = 1000 # hz
        self.dura = np.array([[]]) # 2D arrat segmentation of data to be analyzed
        #T = 1000; # ms
        #hb = 10 # hz
    
    def get_coherence(self, data_1, data_2, return_singalspectrum = False):
        
        #dura_persample = int(round((self.dura[0,1] - self.dura[0,0])/(1/self.Fs*1000)))
        
        dura = np.round(self.dura*self.Fs/1000).astype(int)
        dura_persample = dura[0,1] - dura[0,0]
    
        nw = int(round(dura_persample*self.hW_real/self.Fs))
        ntap = int(np.floor(2*nw - 1))
        win_dpss = windows.dpss(dura_persample, nw, ntap, sym=False)

        n_data = dura.shape[0]
        if dura_persample%2 == 0: fft_len = int(round(dura_persample/2)) + 1
        else: fft_len = int(round((dura_persample+1)/2))


        
        if return_singalspectrum:
            s12_all = np.zeros([n_data, fft_len], dtype=complex)
            s1_all = np.zeros([n_data, fft_len], dtype=float)
            s2_all = np.zeros([n_data, fft_len], dtype=float)
        else:
            s12 = np.zeros(fft_len, dtype=complex)
            s1 = np.zeros(fft_len, dtype=float)
            s2 = np.zeros(fft_len, dtype=float)
        
        
        for dura_i in dura:
            #print(d1.shape,d2.shape)
            tap_data_1 = win_dpss*data_1[dura_i[0]:dura_i[1]]
            tap_data_2 = win_dpss*data_2[dura_i[0]:dura_i[1]]
            
            coef_1 = np.fft.rfft(tap_data_1, axis = -1)
            coef_2 = np.fft.rfft(tap_data_2, axis = -1)
            
            if return_singalspectrum:
                s12_all[dura_i] = np.mean(coef_1 * np.conjugate(coef_2), 0)
                s1_all[dura_i] = np.mean(np.abs(coef_1)**2, 0)
                s2_all[dura_i] = np.mean(np.abs(coef_2)**2, 0)
            else:
                s12 += np.mean(coef_1 * np.conjugate(coef_2), 0)/n_data
                s1 += np.mean(np.abs(coef_1)**2, 0)/n_data
                s2 += np.mean(np.abs(coef_2)**2, 0)/n_data
        
        if return_singalspectrum:
            s12 = s12_all.mean(0)
            s1 = s1_all.mean(0)
            s2 = s2_all.mean(0)
        
        cohe = np.abs(s12/np.sqrt(s1*s2))
        phase = np.angle(s12/np.sqrt(s1*s2))

        if return_singalspectrum:
            R = mydata.mydata()
            R.s1 = s1
            R.s2 = s2
            R.s12 = s12
            R.s12_all = s12_all
            R.s1_all = s1_all
            R.s2_all = s2_all
            R.cohe = cohe
            R.phase = phase
            
            return R
        else:
            R = mydata.mydata()
            R.s1 = s1
            R.s2 = s2
            R.s12 = s12
            R.cohe = cohe
            R.phase = phase
            
            return R
#%%
tapcoh = taper_coherence_multisignal()
tapcoh.dura = stim_onoff_new
coh = tapcoh.get_coherence(mua_1, mua_2)



tapcoh.dura = stim_onoff_new_att
coh_att = tapcoh.get_coherence(mua_1, mua_2)
#%%
tapcoh = taper_coherence_multisignal()
tapcoh.hW_real = 3
tapcoh.dura = stim_onoff_new
coh = tapcoh.get_coherence(mua_1, mua_2)


tapcoh.hW_real = 3
tapcoh.dura = stim_onoff_new_att
coh_att = tapcoh.get_coherence(mua_1, mua_2)

#%%
fig, ax = plt.subplots(2,1)
ax[0].plot(Fs*np.arange(coh.cohe.shape)/(2*coh.cohe.shape[0]), coh.cohe)
ax[0].plot(Fs*np.arange(coh.cohe.shape)/(2*coh.cohe.shape[0]), coh_att.cohe)

ax[1].plot(Fs*np.arange(coh.cohe.shape)/(2*coh.cohe.shape[0]), coh.phase)
ax[1].plot(Fs*np.arange(coh.cohe.shape)/(2*coh.cohe.shape[0]), coh_att.phase)
