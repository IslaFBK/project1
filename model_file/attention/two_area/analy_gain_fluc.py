#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:28:25 2021

@author: shni2598
"""

import matplotlib as mpl
mpl.use('Agg')
import scipy.stats
#import load_data_dict
import mydata
import brian2.numpy_ as np
#from brian2.only import *
#import post_analysis as psa
#import firing_rate_analysis as fra
#import frequency_analysis as fqa
#import fano_mean_match
#import find_change_pts
import connection as cn
#import pickle
import sys
#import os
import matplotlib.pyplot as plt
#import shutil

#from scipy import optimize

import gain_fluctuation as gfluc

#%%
data_dir = 'raw_data/'
analy_type = 'fbrg' 
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/state/param1/'+data_dir

sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#loop_num = 0

#good_dir = 'good/'
#goodsize_dir = 'good_size/'

savefile_name = 'data_anly_gain' #'data_anly' data_anly_temp 'data_anly_onoff'
save_apd = ''

#onff_method = 'threshold'

#thre_spon = 5#4
#thre_stim = [20] #[12, 15, 30]

fftplot = 1; getfano = 1
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1
firing_rate_long = 1

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 0
else: get_ani = 0

save_analy_file = True

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']


data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

if analy_type == 'fbrg': # fbrg: feedback range
    title = 'hz_2irie%.2f_2ndgk%.1f_2e1ir%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                               data.inter.param.tau_p_d_e2_i1)


n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = 200*2**np.arange(n_StimAmp) # [400]

#%%
#mua_loca = [0, 0]
mua_range = 2 
#mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
record_neugroup = []
elec_posi_x = np.arange(-12, 12.1, 4)
elec_posi = np.zeros([elec_posi_x.shape[0], 2])
elec_posi[:,0] = elec_posi_x
for elec in elec_posi:
    record_neugroup.append(cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, elec, mua_range, data.a1.param.width))

#%%
simu_time_tot = data.param.simutime
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
#%%

stim_indptr = np.array([0,50,100,150])

MUA_addi_noatt, MUA_predict_addi_noatt = gfluc.xvalid_addititive_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

MUA_mutp_noatt, MUA_predict_mutp_noatt = gfluc.xvalid_multiplicative_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

MUA_affi_noatt, MUA_predict_affi_noatt = gfluc.xvalid_affine_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)
#stim_indptr = np.array([0,50,100,])
#stim_indptr = np.array([0,50,100,150])

MUA_indep_noatt, MUA_predict_indep_noatt = gfluc.xvalid_indep(data.a1.ge.spk_matrix, \
                                            data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

#%%
q_addi_noatt = gfluc.quality_index(MUA_predict_addi_noatt, MUA_addi_noatt, MUA_predict_indep_noatt, MUA_indep_noatt)
#print(q_addi)

q_mutp_noatt = gfluc.quality_index(MUA_predict_mutp_noatt, MUA_mutp_noatt, MUA_predict_indep_noatt, MUA_indep_noatt)
#print(q_mutp)

q_affi_noatt = gfluc.quality_index(MUA_predict_affi_noatt, MUA_affi_noatt, MUA_predict_indep_noatt, MUA_indep_noatt)
#print(q_affi)

data_anly.addi = mydata.mydata()
data_anly.multip = mydata.mydata()
data_anly.affine = mydata.mydata()

data_anly.addi.q_index_noatt = q_addi_noatt
data_anly.multip.q_index_noatt = q_mutp_noatt
data_anly.affine.q_index_noatt = q_affi_noatt

data_anly.elec_posi_x = elec_posi_x 
data_anly.record_neugroup = record_neugroup
#%%

MUA_addi_att, MUA_predict_addi_att = gfluc.xvalid_addititive_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:].copy(), record_neugroup, stim_indptr)

MUA_mutp_att, MUA_predict_mutp_att = gfluc.xvalid_multiplicative_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:].copy(), record_neugroup, stim_indptr)

MUA_affi_att, MUA_predict_affi_att = gfluc.xvalid_affine_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:].copy(), record_neugroup, stim_indptr)
#stim_indptr = np.array([0,50,100,])
#stim_indptr = np.array([0,50,100,150])

MUA_indep_att, MUA_predict_indep_att = gfluc.xvalid_indep(data.a1.ge.spk_matrix, \
                                            data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:].copy(), record_neugroup, stim_indptr)

#%%
q_addi_att = gfluc.quality_index(MUA_predict_addi_att, MUA_addi_att, MUA_predict_indep_att, MUA_indep_att)
#print(q_addi)

q_mutp_att = gfluc.quality_index(MUA_predict_mutp_att, MUA_mutp_att, MUA_predict_indep_att, MUA_indep_att)
#print(q_mutp)

q_affi_att = gfluc.quality_index(MUA_predict_affi_att, MUA_affi_att, MUA_predict_indep_att, MUA_indep_att)
#print(q_affi)

# data_anly.addi = mydata.mydata()
# data_anly.multip = mydata.mydata()
# data_anly.affine = mydata.mydata()

data_anly.addi.q_index_att = q_addi_att
data_anly.multip.q_index_att = q_mutp_att
data_anly.affine.q_index_att = q_affi_att

#%%
R_addi = gfluc.get_addititive_fluc(data.a1.ge.spk_matrix, \
                             data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), \
                                 record_neugroup, \
                                     stim_indptr)

data_anly.addi.coup_noatt = R_addi.coup_addi
data_anly.addi.fluc_noatt = R_addi.fluc_addi
data_anly.addi.mean_MUA_noatt = R_addi.mean_MUA
data_anly.addi.stim_indptr_noatt = R_addi.stim_indptr

R_multip = gfluc.get_multiplicative_fluc(data.a1.ge.spk_matrix, \
                             data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), \
                                 record_neugroup, \
                                     stim_indptr)
    
data_anly.multip.coup_noatt = R_multip.coup_multip
data_anly.multip.fluc_noatt = R_multip.fluc_multip
data_anly.multip.stim_indptr_noatt = R_multip.stim_indptr

R_aff = gfluc.get_affine_fluc(data.a1.ge.spk_matrix, \
                             data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), \
                                 record_neugroup, \
                                     stim_indptr, trialforSolveAmbiguity=0)

data_anly.affine.coup_multip_noatt = R_aff.coup_multip
data_anly.affine.fluc_multip_noatt = R_aff.fluc_multip
data_anly.affine.coup_addi_noatt = R_aff.coup_addi
data_anly.affine.fluc_addi_noatt = R_aff.fluc_addi
data_anly.affine.stim_indptr_noatt = R_aff.stim_indptr

data_anly.MUA_noatt = R_aff.MUA_multi
#%%
R_addi = gfluc.get_addititive_fluc(data.a1.ge.spk_matrix, \
                             data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:].copy(), \
                                 record_neugroup, \
                                     stim_indptr)

data_anly.addi.coup_att = R_addi.coup_addi
data_anly.addi.fluc_att = R_addi.fluc_addi
data_anly.addi.mean_MUA_att = R_addi.mean_MUA
data_anly.addi.stim_indptr_att = R_addi.stim_indptr

R_multip = gfluc.get_multiplicative_fluc(data.a1.ge.spk_matrix, \
                             data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:].copy(), \
                                 record_neugroup, \
                                     stim_indptr)
    
data_anly.multip.coup_att = R_multip.coup_multip
data_anly.multip.fluc_att = R_multip.fluc_multip
data_anly.multip.stim_indptr_att = R_multip.stim_indptr

R_aff = gfluc.get_affine_fluc(data.a1.ge.spk_matrix, \
                             data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:].copy(), \
                                 record_neugroup, \
                                     stim_indptr, trialforSolveAmbiguity=0)

data_anly.affine.coup_multip_att = R_aff.coup_multip
data_anly.affine.fluc_multip_att = R_aff.fluc_multip
data_anly.affine.coup_addi_att = R_aff.coup_addi
data_anly.affine.fluc_addi_att = R_aff.fluc_addi
data_anly.affine.stim_indptr_att = R_aff.stim_indptr

data_anly.MUA_att = R_aff.MUA_multi
#%%

fig, ax = plt.subplots(1,3, figsize=[10,5])
for st in range(len(stim_amp)):    
    ax[0].plot(elec_posi[:,0], data_anly.addi.mean_MUA_noatt[:,st], ls='--', c=clr[st], label='addi_MUA;%dhz'%stim_amp[st])
    ax[0].plot(elec_posi[:,0], data_anly.addi.mean_MUA_att[:,st], ls='-', c=clr[st])#, label='addi_MUA;%dhz'%stim_amp[st])

    ax[1].errorbar(elec_posi[:,0], data_anly.multip.coup_noatt[:,st], \
                   data_anly.multip.fluc_noatt[st*n_perStimAmp:(st+1)*n_perStimAmp].std()*np.ones(len(elec_posi)),\
                   ls='--', c=clr[st], label='multi coup&fluc;%dhz'%stim_amp[st])
    ax[1].errorbar(elec_posi[:,0], data_anly.multip.coup_att[:,st], \
                   data_anly.multip.fluc_att[st*n_perStimAmp:(st+1)*n_perStimAmp].std()*np.ones(len(elec_posi)),\
                   ls='-', c=clr[st])#, label='multi coup&fluc;%dhz'%stim_amp[st])

    ax[2].errorbar(elec_posi[:,0], data_anly.affine.coup_multip_noatt[:,st], \
                   data_anly.affine.fluc_multip_noatt[st*n_perStimAmp:(st+1)*n_perStimAmp].std()*np.ones(len(elec_posi)),\
                   ls='--', c=clr[st], label='aff-multi coup&fluc;%dhz'%stim_amp[st])
    ax[2].errorbar(elec_posi[:,0], data_anly.affine.coup_multip_att[:,st], \
                   data_anly.affine.fluc_multip_att[st*n_perStimAmp:(st+1)*n_perStimAmp].std()*np.ones(len(elec_posi)),\
                   ls='-', c=clr[st])#, label='aff-multi coup&fluc;%dhz'%stim_amp[st])


ax[2].errorbar(elec_posi[:,0], data_anly.affine.coup_addi_noatt, data_anly.affine.fluc_addi_noatt.std()*np.ones(elec_posi.shape[0]), \
               ls='--', c=clr[st+1], label='aff-addi coup&fluc')#+data_anly.addi.coup_noatt)
ax[2].errorbar(elec_posi[:,0], data_anly.affine.coup_addi_att, data_anly.affine.fluc_addi_att.std()*np.ones(elec_posi.shape[0]), \
               ls='-', c=clr[st+1])#, label='aff-addi coup&fluc')#+data_anly.addi.coup_noatt)

ax[0].errorbar(elec_posi[:,0], data_anly.addi.coup_noatt, data_anly.addi.fluc_noatt.std()*np.ones(elec_posi.shape[0]), \
               ls='--', c=clr[st+1], label='addi coup&fluc')#+data_anly.addi.coup_noatt)
ax[0].errorbar(elec_posi[:,0], data_anly.addi.coup_att, data_anly.addi.fluc_att.std()*np.ones(elec_posi.shape[0]), \
               ls='-', c=clr[st+1])#, label='addi coup&fluc')#+data_anly.addi.coup_noatt)

for axx in ax:
    axx.legend()
#for st in range(len(stim_amp)):    
title_ = title + '\n_gain_coup_fluc'#%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
fig.suptitle(title_)
savetitle = title_.replace('\n','')
savetitle = savetitle+'_%d'%(loop_num)+'.png'
if save_img: fig.savefig(savetitle)
plt.close()

#%%
fig, ax = plt.subplots(2,3, figsize=[10,8])

#for axy in ax:
ax[0,0].scatter(data_anly.multip.q_index_noatt, data_anly.addi.q_index_noatt)
ax[0,1].scatter(data_anly.affine.q_index_noatt, data_anly.addi.q_index_noatt)
ax[0,2].scatter(data_anly.affine.q_index_noatt, data_anly.multip.q_index_noatt)
ax[1,0].scatter(data_anly.multip.q_index_att, data_anly.addi.q_index_att)
ax[1,1].scatter(data_anly.affine.q_index_att, data_anly.addi.q_index_att)
ax[1,2].scatter(data_anly.affine.q_index_att, data_anly.multip.q_index_att)

for axy in ax:
    for axxi, axx in enumerate(axy):
        if axxi == 0: axx.set_xlabel('multip'); axx.set_ylabel('addi'); 
        elif axxi == 1: axx.set_xlabel('affine'); axx.set_ylabel('addi'); 
        elif axxi == 2: axx.set_xlabel('affine'); axx.set_ylabel('multip'); 
            
        axx.plot([-0.2,1.1],[-0.2,1.1],'r--')
        axx.set_xlim([-0.2,1.1])
        axx.set_ylim([-0.2,1.1])

title_ = title + '\n_quality_index'#%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
fig.suptitle(title_)
savetitle = title_.replace('\n','')
savetitle = savetitle+'_%d'%(loop_num)+'.png'
if save_img: fig.savefig(savetitle)
plt.close()
#%%
data_anly.addi.fluc_std = np.zeros([n_StimAmp, 2])
data_anly.multip.fluc_std = np.zeros([n_StimAmp, 2])
data_anly.affine.fluc_multip_std = np.zeros([n_StimAmp, 2])
data_anly.affine.fluc_addi_std = np.zeros([n_StimAmp, 2])

for st in range(len(stim_amp)): 
    data_anly.addi.fluc_std[st,0] = data_anly.addi.fluc_noatt[st*n_perStimAmp:(st+1)*n_perStimAmp].std()
    data_anly.addi.fluc_std[st,1] = data_anly.addi.fluc_att[st*n_perStimAmp:(st+1)*n_perStimAmp].std()
    
    data_anly.multip.fluc_std[st,0] = data_anly.multip.fluc_noatt[st*n_perStimAmp:(st+1)*n_perStimAmp].std()
    data_anly.multip.fluc_std[st,1] = data_anly.multip.fluc_att[st*n_perStimAmp:(st+1)*n_perStimAmp].std()
    
    data_anly.affine.fluc_multip_std[st,0] = data_anly.affine.fluc_multip_noatt[st*n_perStimAmp:(st+1)*n_perStimAmp].std()
    data_anly.affine.fluc_multip_std[st,1] = data_anly.affine.fluc_multip_att[st*n_perStimAmp:(st+1)*n_perStimAmp].std()
    
    data_anly.affine.fluc_addi_std[st,0] = data_anly.affine.fluc_addi_noatt[st*n_perStimAmp:(st+1)*n_perStimAmp].std()
    data_anly.affine.fluc_addi_std[st,1] = data_anly.affine.fluc_addi_att[st*n_perStimAmp:(st+1)*n_perStimAmp].std()
    
#%%
fig, ax = plt.subplots(4,1, figsize=[6,8])
for st in range(len(data_anly.addi.fluc_std)):
#for ii, std in enumerate(data_anly.addi.fluc_std):
    ax[0].plot(np.arange(2)+st*2, data_anly.addi.fluc_std[st], label='st:%d'%stim_amp[st])
    ax[1].plot(np.arange(2)+st*2, data_anly.multip.fluc_std[st])
    ax[2].plot(np.arange(2)+st*2, data_anly.affine.fluc_multip_std[st])
    ax[3].plot(np.arange(2)+st*2, data_anly.affine.fluc_addi_std[st])
    
ax[0].set_title('addi'); ax[0].xaxis.set_visible(False)
ax[1].set_title('multip'); ax[1].xaxis.set_visible(False)
ax[2].set_title('affine multip'); ax[2].xaxis.set_visible(False)
ax[3].set_title('affine addi'); ax[3].xaxis.set_visible(False)

ax[0].legend()

title_ = title + '\n_fluc_modu'#%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
fig.suptitle(title_)
savetitle = title_.replace('\n','')
savetitle = savetitle+'_%d'%(loop_num)+'.png'
if save_img: fig.savefig(savetitle)
plt.close()
#%%
if save_analy_file:
    data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)

