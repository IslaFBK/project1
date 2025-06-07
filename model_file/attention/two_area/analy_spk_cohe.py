#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 22:35:06 2021

@author: shni2598
"""


#%%
import firing_rate_analysis as fra
import frequency_analysis as fqa
import spk_cohe
import mydata
import brian2.numpy_ as np
import matplotlib as mpl
mpl.use('Agg')
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt

#%%
data_dir = 'raw_data/'
analy_type = 'fffbrg'
ref_sig = 'MUA' #LFP MUA
area = 'a2'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/state/param1/stim2/longstim/raw_data/'
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly_coh_mua_a2' #'data_anly' data_anly_temp data_anly_nc_normal
save_apd = '_mua_a2'

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 0
else: get_ani = 0

save_analy_file = True

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

# def spike_mua_coherence
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

if analy_type == 'fbrg': # fbrg: feedback range
    title = 'hz_2irie%.2f_2ndgk%.1f_2e1ir%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                               data.inter.param.tau_p_d_e2_i1)
if analy_type == 'fffbrg':
    title = '1e2e%.2f_2e1e%.2f'%(data.inter.param.w_e1_e2_mean/5, \
                                               data.inter.param.w_e2_e1_mean/5)


#%%
n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
#stim_amp = 200*2**np.arange(n_StimAmp)
stim_amp = np.array([400])

#%%
if ref_sig == 'MUA':
    mua_loca = [0, 0]
    mua_range = 5 
    mua_neuron_noatt = cn.findnearbyneuron.findnearbyneuron(data.__dict__[area].param.e_lattice, mua_loca, mua_range, data.__dict__[area].param.width)
    mua_neuron_att = cn.findnearbyneuron.findnearbyneuron(data.__dict__[area].param.e_lattice, mua_loca, mua_range, data.__dict__[area].param.width)
    mua_loca = [-32, -32]
    mua_range = 5 
    mua_neuron_unatt = cn.findnearbyneuron.findnearbyneuron(data.__dict__[area].param.e_lattice, mua_loca, mua_range, data.__dict__[area].param.width)

spk_loca = [0, 0]
spk_range = 2 
spk_neuron_noatt = cn.findnearbyneuron.findnearbyneuron(data.__dict__[area].param.e_lattice, spk_loca, spk_range, data.__dict__[area].param.width)
spk_neuron_att = cn.findnearbyneuron.findnearbyneuron(data.__dict__[area].param.e_lattice, spk_loca, spk_range, data.__dict__[area].param.width)
spk_loca = [-32, -32]
spk_range = 2 
spk_neuron_unatt = cn.findnearbyneuron.findnearbyneuron(data.__dict__[area].param.e_lattice, spk_loca, spk_range, data.__dict__[area].param.width)

#%%
simu_time_tot = data.param.simutime
data.__dict__[area].ge.get_sparse_spk_matrix([data.__dict__[area].param.Ne, simu_time_tot*10], 'csc')

if ref_sig == 'MUA':
    mua_mat_noatt = data.__dict__[area].ge.spk_matrix[mua_neuron_noatt]
    mua_mat_unatt = data.__dict__[area].ge.spk_matrix[mua_neuron_unatt]
    mua_mat_att = data.__dict__[area].ge.spk_matrix[mua_neuron_att]

spk_mat_noatt = data.__dict__[area].ge.spk_matrix[spk_neuron_noatt]
spk_mat_unatt = data.__dict__[area].ge.spk_matrix[spk_neuron_unatt]
spk_mat_att = data.__dict__[area].ge.spk_matrix[spk_neuron_att]

#%%
data_anly.cohe_noatt = []
data_anly.cohe_unatt = []
data_anly.cohe_att = []

fig, ax = plt.subplots(3, n_StimAmp, figsize=[4*n_StimAmp, 10])
for st in range(n_StimAmp):
    
    dura_noatt = data.__dict__['a1'].param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
    dura_unatt = data.__dict__['a1'].param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
    dura_att = data.__dict__['a1'].param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
    if ref_sig == 'MUA':
        R_noatt = spk_cohe.spk_mua_coherence(spk_mat_noatt, mua_mat_noatt, dura_noatt, discard_init = 200, hfwin_mua_seg=150, dt = 0.1)
        R_unatt = spk_cohe.spk_mua_coherence(spk_mat_unatt, mua_mat_unatt, dura_unatt, discard_init = 200, hfwin_mua_seg=150, dt = 0.1)
        R_att = spk_cohe.spk_mua_coherence(spk_mat_att, mua_mat_att, dura_att, discard_init = 200, hfwin_mua_seg=150, dt = 0.1)
    if ref_sig == 'LFP':
        R_noatt = spk_cohe.spk_lfp_coherence(spk_mat_noatt, data.__dict__[area].ge.LFP[0], dura_noatt, discard_init = 200, hfwin_lfp_seg=150, dt = 0.1)        
        R_unatt = spk_cohe.spk_lfp_coherence(spk_mat_unatt, data.__dict__[area].ge.LFP[1], dura_unatt, discard_init = 200, hfwin_lfp_seg=150, dt = 0.1)
        R_att = spk_cohe.spk_lfp_coherence(spk_mat_att, data.__dict__[area].ge.LFP[0], dura_att, discard_init = 200, hfwin_lfp_seg=150, dt = 0.1)

    data_anly.cohe_noatt.append(R_noatt)
    data_anly.cohe_unatt.append(R_unatt)
    data_anly.cohe_att.append(R_att)
    
    plt_len = (R_noatt.freq <= 150).sum()
    
    if n_StimAmp == 1:
        ax[0].plot(R_noatt.freq[:plt_len], R_noatt.cohe[:plt_len], label='cohe,noatt;%dHz'%stim_amp[st])
        ax[0].plot(R_unatt.freq[:plt_len], R_unatt.cohe[:plt_len], label='cohe,unatt;%dHz'%stim_amp[st])
        ax[0].plot(R_att.freq[:plt_len], R_att.cohe[:plt_len], label='cohe,att;%dHz'%stim_amp[st])
    
        ax[1].loglog(R_noatt.freq[1:plt_len], R_noatt.staRef_pw[1:plt_len], label='sta'+ref_sig+'_pw,noatt;%dHz'%stim_amp[st])
        ax[1].loglog(R_unatt.freq[1:plt_len], R_unatt.staRef_pw[1:plt_len], label='sta'+ref_sig+'_pw,unatt;%dHz'%stim_amp[st])
        ax[1].loglog(R_att.freq[1:plt_len], R_att.staRef_pw[1:plt_len], label='sta'+ref_sig+'_pw,att;%dHz'%stim_amp[st])
    
        ax[2].plot(np.arange(R_noatt.staRef.shape[0])/10, R_noatt.staRef, label='sta'+ref_sig+',noatt;%dHz'%stim_amp[st])
        ax[2].plot(np.arange(R_unatt.staRef.shape[0])/10, R_unatt.staRef, label='sta'+ref_sig+',unatt;%dHz'%stim_amp[st])
        ax[2].plot(np.arange(R_att.staRef.shape[0])/10, R_att.staRef, label='sta'+ref_sig+',att;%dHz'%stim_amp[st])
    


    else:
        ax[0,st].plot(R_noatt.freq[:plt_len], R_noatt.cohe[:plt_len], label='cohe,noatt;%dHz'%stim_amp[st])
        ax[0,st].plot(R_unatt.freq[:plt_len], R_unatt.cohe[:plt_len], label='cohe,unatt;%dHz'%stim_amp[st])
        ax[0,st].plot(R_att.freq[:plt_len], R_att.cohe[:plt_len], label='cohe,att;%dHz'%stim_amp[st])
    
        ax[1,st].loglog(R_noatt.freq[1:plt_len], R_noatt.staRef_pw[1:plt_len], label='sta'+ref_sig+'_pw,noatt;%dHz'%stim_amp[st])
        ax[1,st].loglog(R_unatt.freq[1:plt_len], R_unatt.staRef_pw[1:plt_len], label='sta'+ref_sig+'_pw,unatt;%dHz'%stim_amp[st])
        ax[1,st].loglog(R_att.freq[1:plt_len], R_att.staRef_pw[1:plt_len], label='sta'+ref_sig+'_pw,att;%dHz'%stim_amp[st])
    
        ax[2,st].plot(np.arange(R_noatt.staRef.shape[0])/10, R_noatt.staRef, label='sta'+ref_sig+',noatt;%dHz'%stim_amp[st])
        ax[2,st].plot(np.arange(R_unatt.staRef.shape[0])/10, R_unatt.staRef, label='sta'+ref_sig+',unatt;%dHz'%stim_amp[st])
        ax[2,st].plot(np.arange(R_att.staRef.shape[0])/10, R_att.staRef, label='sta'+ref_sig+',att;%dHz'%stim_amp[st])
    
    if n_StimAmp == 1:
        for axy in ax:
            axy.legend()  
    else:
        for axy in ax:
            for axx in axy:
                axx.legend()

title_coh = title + '_coh'
fig.suptitle(title_coh)
savetitle = title_coh.replace('\n','')
savetitle = savetitle+save_apd+'_%d'%(loop_num)+'.png'
if save_img: fig.savefig(savetitle)
plt.close()


#%%
if save_analy_file:
    data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)

