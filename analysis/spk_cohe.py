#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:55:34 2021

@author: shni2598
"""

#%%
import firing_rate_analysis as fra
import frequency_analysis as fqa
import mydata
import numpy as np

#%%
def spk_mua_coherence(spk_mat, mua_mat, dura, discard_init = 200, hfwin_mua_seg=150, dt = 0.1):
    """
    Methods from "Modulation of Oscillatory Neuronal Synchronization by Selective Visual Attention"
    Use MUA instead of LFP
    Parameters
    ----------
    spk_mat : TYPE
        spike sparse matrix.
    mua_mat : TYPE
        mua sparse matrix.
    dura : TYPE
        2D array; start and end of analysis period
    discard_init : TYPE, optional
        ms; length of initial period of MUA to be discraded ; The default is 200 ms.
    hfwin_mua_seg : TYPE, optional
        ms; length of half mua segment The default is 150 ms.
    dt : TYPE, optional
        ms; sample interval. The default is 0.1 ms.

    Returns
    -------
    None.

    """
    #discard_init = 200
    #hfwin_mua_seg = 150
    #dt = 0.1
    dt_1 = int(round(1/dt))



    discard_init = int(round(discard_init/dt))
    hfwin_mua_seg = int(round(hfwin_mua_seg/dt))

    total_spk = 0
    #i = 0
    
    for dura_t in dura:
        #print(total_spk)
        #print(i)
        #i+=1
        mua = fra.get_spkcount_sum_sparmat(mua_mat, start_time=dura_t[0], end_time=dura_t[1],\
                       sample_interval = dt,  window = 3, dt = dt)
        
        spk_t = spk_mat[:, dura_t[0]*dt_1:dura_t[1]*dt_1].nonzero()[1]
        
        spk_t = spk_t[(spk_t > (discard_init + hfwin_mua_seg)) & (spk_t < (mua.shape[0] - hfwin_mua_seg))]
        #print(spk_t.shape)
        for spk in spk_t:
            total_spk += 1
            mua_i = mua[spk-hfwin_mua_seg:spk+hfwin_mua_seg]
            coef = np.fft.rfft(mua_i - mua_i.mean())/(2*hfwin_mua_seg)
            if total_spk == 1:
                
                stMua_pw = np.zeros(coef.shape)
                staMua_coef = np.zeros(coef.shape, dtype=complex)
                staMua = np.zeros(mua_i.shape)
            #print(coef.shape)
            stMua_pw += np.abs(coef)**2
            staMua_coef += coef
            staMua += mua_i
    
    stMua_pw /= total_spk
    staMua_coef /= total_spk
    staMua /= total_spk
    
    cohe = np.abs(staMua_coef)**2/stMua_pw
    freq = np.fft.rfftfreq(hfwin_mua_seg*2, d=dt/1000)
    power, _ = fqa.myfft(staMua-staMua.mean(), Fs=dt_1*1000, power=True)
    
    R = mydata.mydata()
    R.cohe = cohe
    R.freq = freq
    R.staRef_pw = power # Ref: MUA
    R.staRef = staMua
    
    return R #cohe, power, total_mua_seg, freq

#%%
# mua_loca = [0, 0]
# mua_range = 5 
# mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)

# spk_loca = [0, 0]
# spk_range = 2 
# spk_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, spk_loca, spk_range, data.a1.param.width)
# #%%
# simu_time_tot = data.param.simutime
# data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10], 'csc')

# mua_mat = data.a1.ge.spk_matrix[mua_neuron]
# spk_mat = data.a1.ge.spk_matrix[spk_neuron]

# #%%
# for st in range(n_StimAmp):
    
#     dura_noatt = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
#     dura_att = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
    
# # #%%
# # cohe_noatt, mua_pow_noatt, mua_sta_noatt, freq = spk_mua_coherence(spk_mat, mua_mat, dura_noatt, discard_init = 200, hfwin_mua_seg=150, dt = 0.1)
# # cohe_att, mua_pow_att, mua_sta_att, freq = spk_mua_coherence(spk_mat, mua_mat, dura_att, discard_init = 200, hfwin_mua_seg=150, dt = 0.1)
# #%%
# R_noatt = spk_mua_coherence(spk_mat, mua_mat, dura_noatt, discard_init = 200, hfwin_mua_seg=150, dt = 0.1)
# R_att = spk_mua_coherence(spk_mat, mua_mat, dura_att, discard_init = 200, hfwin_mua_seg=150, dt = 0.1)
#%%
def spk_lfp_coherence(spk_mat, lfp, dura, discard_init = 200, hfwin_lfp_seg=150, dt = 0.1):
    """
    Methods from "Modulation of Oscillatory Neuronal Synchronization by Selective Visual Attention"
    Parameters
    ----------
    spk_mat : TYPE
        spike sparse matrix.
    lfp : TYPE
        LFP signal, 1D array.
    dura : TYPE
        2D array; start and end of analysis period
    discard_init : TYPE, optional
        ms; length of initial period of LFP to be discraded ; The default is 200 ms.
    hfwin_lfp_seg : TYPE, optional
        ms; length of half LFP segment The default is 150 ms.
    dt : TYPE, optional
        ms; sample interval. The default is 0.1 ms.

    Returns
    -------
    None.

    """
    #discard_init = 200
    #hfwin_lfp_seg = 150
    #dt = 0.1
    dt_1 = int(round(1/dt))



    discard_init = int(round(discard_init/dt))
    hfwin_lfp_seg = int(round(hfwin_lfp_seg/dt))

    total_spk = 0
    #i = 0
    
    for dura_t in dura:
        #print(total_spk)
        #print(i)
        #i+=1
        # mua = fra.get_spkcount_sum_sparmat(mua_mat, start_time=dura_t[0], end_time=dura_t[1],\
        #                sample_interval = dt,  window = 3, dt = dt)
        lfp_tri = lfp[dura_t[0]*dt_1:dura_t[1]*dt_1]
        
        spk_t = spk_mat[:, dura_t[0]*dt_1:dura_t[1]*dt_1].nonzero()[1]
        
        spk_t = spk_t[(spk_t > (discard_init + hfwin_lfp_seg)) & (spk_t < (lfp_tri.shape[0] - hfwin_lfp_seg))]
        #print(spk_t.shape)
        for spk in spk_t:
            total_spk += 1
            lfp_i = lfp_tri[spk-hfwin_lfp_seg:spk+hfwin_lfp_seg]
            coef = np.fft.rfft(lfp_i - lfp_i.mean())/(2*hfwin_lfp_seg)
            if total_spk == 1:
                
                stLfp_pw = np.zeros(coef.shape)
                staLfp_coef = np.zeros(coef.shape, dtype=complex)
                staLfp = np.zeros(lfp_i.shape)
            #print(coef.shape)
            stLfp_pw += np.abs(coef)**2
            staLfp_coef += coef
            staLfp += lfp_i
    
    stLfp_pw /= total_spk
    staLfp_coef /= total_spk
    staLfp /= total_spk
    
    cohe = np.abs(staLfp_coef)**2/stLfp_pw
    freq = np.fft.rfftfreq(hfwin_lfp_seg*2, d=dt/1000)
    power, _ = fqa.myfft(staLfp-staLfp.mean(), Fs=dt_1*1000, power=True)
    
    R = mydata.mydata()
    R.cohe = cohe
    R.freq = freq
    R.staRef_pw = power # Ref: LFP
    R.staRef = staLfp
    
    return R #cohe, power, total_mua_seg, freq


