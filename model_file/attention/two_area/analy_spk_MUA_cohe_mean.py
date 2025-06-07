#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:48:22 2021

@author: shni2598
"""

import matplotlib as mpl
mpl.use('Agg')
import scipy.stats
#import load_data_dict
import mydata
import brian2.numpy_ as np
from scipy.stats import sem

#from brian2.only import *
#import post_analysis as psa
#import firing_rate_analysis as fra
#import frequency_analysis as fqa
#import fano_mean_match
#import find_change_pts
import connection as cn
#import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
data_dir = 'raw_data/'
analy_type = 'fbrg'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/'+data_dir
#sys_argv = int(sys.argv[1])
#rep_ind*20 + ie_num
#loop_num = 0

good_dir = 'good/'
goodsize_dir = 'good_size/'

#savefile_name = 'data_anly' #'data_anly' data_anly_temp
save_dir = 'mean_results/'
data_analy_file = 'data_anly_coh'#  data_anly_onoff_thres  data_anly_onoff_thres_samesens data_anly_onoff_hmm data_anly_thr data_anly' #data_anly_fano data_anly_fano10hz data_anly_nc data_anly_fano10hz data_anly_fanorange
save_apd = '' #'_thr'
#save_apd_list=['_sens_thre', '_asso_thre'] # ['_sens_thre_samesens', '_asso_thre_samesens']
#save_apd_list=['_sens_hmm', '_asso_hmm'] # ['_sens_thre_samesens', '_asso_thre_samesens']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# fftplot = 1; getfano = 1; 
# get_nscorr = 1; get_nscorr_t = 1
# get_TunningCurve = 1; get_HzTemp = 1;

#get_intvOn_var = 0
save_img = 1
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

loop_num = 0 
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = [400] #200*2**np.arange(n_StimAmp)

# mua_loca = [0, 0]
# mua_range = 5 
# mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)

#mua_win = 10
        
#%%
#n_reliz = 20

#%%
n_param = 2
repeat = 20

for param_id in range(n_param):
    
    cohe_noatt = [[] for _ in range(n_StimAmp)]
    staMUA_pw_noatt = [[] for _ in range(n_StimAmp)]
    staMUA_noatt = [[] for _ in range(n_StimAmp)]
    
    cohe_att = [[] for _ in range(n_StimAmp)]
    staMUA_pw_att = [[] for _ in range(n_StimAmp)]
    staMUA_att = [[] for _ in range(n_StimAmp)]
    
    for loop_num in range(param_id*repeat,(param_id+1)*repeat):
        
        try: 
            data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        except EOFError:
            print('EOFError: file: %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        
        # for aid, area in enumerate(['onoff_sens','onoff_asso']):
        #     on_t_spon[aid].append(np.concatenate(data_anly.__dict__[area].spon.on_t))
        #     off_t_spon[aid].append(np.concatenate(data_anly.__dict__[area].spon.off_t))
        #     on_amp_spon[aid].append(np.concatenate(data_anly.__dict__[area].spon.on_amp))
        #     off_amp_spon[aid].append(np.concatenate(data_anly.__dict__[area].spon.off_amp))
            
        for stim_id in range(n_StimAmp):
            
            cohe_noatt[stim_id].append(data_anly.cohe_noatt[stim_id].cohe)
            staMUA_pw_noatt[stim_id].append(data_anly.cohe_noatt[stim_id].staMUA_pw)
            staMUA_noatt[stim_id].append(data_anly.cohe_noatt[stim_id].staMUA)
            
            cohe_att[stim_id].append(data_anly.cohe_att[stim_id].cohe)
            staMUA_pw_att[stim_id].append(data_anly.cohe_att[stim_id].staMUA_pw)
            staMUA_att[stim_id].append(data_anly.cohe_att[stim_id].staMUA)

    for stim_id in range(n_StimAmp):
        
        cohe_noatt[stim_id] = np.array(cohe_noatt[stim_id])
        staMUA_pw_noatt[stim_id] = np.array(staMUA_pw_noatt[stim_id])
        staMUA_noatt[stim_id] = np.array(staMUA_noatt[stim_id])

        cohe_att[stim_id] = np.array(cohe_att[stim_id])
        staMUA_pw_att[stim_id] = np.array(staMUA_pw_att[stim_id])
        staMUA_att[stim_id] = np.array(staMUA_att[stim_id])

#%%
    found = False
    cannotfind = False
    file_id = loop_num
    while not found:
        if file_id < param_id*repeat:
            print('cannot find any data file for param: %d'%param_id)
            cannotfind = True
            break
        try: data.load(datapath+'data%d.file'%(file_id))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+'data%d.file'%(file_id)))
            file_id -= 1
            continue
        found = True
        print('use file: %s'%(datapath+'data%d.file'%(file_id)))
    
    if cannotfind:
        continue
    
    if analy_type == 'fbrgbig4': # fbrg: feedback range
        title = '1irie%.2f_1e2e%.1f_pk2e1e%.2f'%(data.param.ie_r_i1, data.inter.param.w_e1_e2_mean/5, \
                                                   data.inter.param.peak_p_e2_e1)
    if analy_type == 'state': # fbrg: feedback range
        title = 'hz_2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                                   data.inter.param.peak_p_e2_e1)
    if analy_type == 'fbrg': # fbrg: feedback range
        title = 'hz_2irie%.2f_2ndgk%.1f_2e1ir%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                                   data.inter.param.tau_p_d_e2_i1)
    
    fig, ax = plt.subplots(3, n_StimAmp, figsize=[4*n_StimAmp, 10])
    freq = data_anly.cohe_noatt[0].freq    
    plt_len = (freq <= 150).sum()
    for stim_id in range(n_StimAmp):
            
        cohe_noatt_mean = np.nanmean(cohe_noatt[stim_id], 0) #= np.array(cohe_noatt[stim_id])
        cohe_noatt_sem = sem(cohe_noatt[stim_id], 0, nan_policy='omit')
        
        staMUA_pw_noatt_mean = np.nanmean(staMUA_pw_noatt[stim_id], 0) #= np.array(cohe_noatt[stim_id])
        staMUA_pw_noatt_sem = sem(staMUA_pw_noatt[stim_id], 0, nan_policy='omit')

        staMUA_noatt_mean = np.nanmean(staMUA_noatt[stim_id], 0) #= np.array(cohe_noatt[stim_id])
        staMUA_noatt_sem = sem(staMUA_noatt[stim_id], 0, nan_policy='omit')

        cohe_att_mean = np.nanmean(cohe_att[stim_id], 0) #= np.array(cohe_noatt[stim_id])
        cohe_att_sem = sem(cohe_att[stim_id], 0, nan_policy='omit')
        
        staMUA_pw_att_mean = np.nanmean(staMUA_pw_att[stim_id], 0) #= np.array(cohe_noatt[stim_id])
        staMUA_pw_att_sem = sem(staMUA_pw_att[stim_id], 0, nan_policy='omit')

        staMUA_att_mean = np.nanmean(staMUA_att[stim_id], 0) #= np.array(cohe_noatt[stim_id])
        staMUA_att_sem = sem(staMUA_att[stim_id], 0, nan_policy='omit')

        if n_StimAmp == 1:
            ax[0].plot(freq[:plt_len], cohe_noatt_mean[:plt_len], label='cohe,noatt;%dHz'%stim_amp[stim_id])
            ax[0].fill_between(freq[:plt_len], cohe_noatt_mean[:plt_len]-cohe_noatt_sem[stim_id], cohe_noatt_mean[:plt_len]+cohe_noatt_sem[stim_id], \
                         ls='--', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)            
            ax[0].plot(freq[:plt_len], cohe_att_mean[:plt_len], label='cohe,att;%dHz'%stim_amp[stim_id])
            ax[0].fill_between(freq[:plt_len], cohe_att_mean[:plt_len]-cohe_att_sem[stim_id], cohe_att_mean[:plt_len]+cohe_att_sem[stim_id], \
                         ls='--', facecolor=clr[1], edgecolor=clr[1], alpha=0.2)            

            ax[1].plot(freq[1:plt_len], staMUA_pw_noatt_mean[1:plt_len], label='staMUA_pw,noatt;%dHz'%stim_amp[stim_id])
            ax[1].fill_between(freq[1:plt_len], staMUA_pw_noatt_mean[1:plt_len]-staMUA_pw_noatt_sem[1:plt_len], staMUA_pw_noatt_mean[1:plt_len]+staMUA_pw_noatt_sem[1:plt_len], \
                         ls='--', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)            
            ax[1].plot(freq[1:plt_len], staMUA_pw_att_mean[1:plt_len], label='staMUA_pw,att;%dHz'%stim_amp[stim_id])
            ax[1].fill_between(freq[1:plt_len], staMUA_pw_att_mean[1:plt_len]-staMUA_pw_att_sem[1:plt_len], staMUA_pw_att_mean[1:plt_len]+staMUA_pw_att_sem[1:plt_len], \
                         ls='--', facecolor=clr[1], edgecolor=clr[1], alpha=0.2)            
            ax[1].set_yscale('log',base=10)
            ax[1].set_xscale('log',base=10)
        
            ax[2].plot(np.arange(staMUA_noatt_mean.shape[0])/10, staMUA_noatt_mean, label='staMUA,noatt;%dHz'%stim_amp[stim_id])
            ax[2].fill_between(np.arange(staMUA_noatt_mean.shape[0])/10, staMUA_noatt_mean-staMUA_noatt_sem, staMUA_noatt_mean+staMUA_noatt_sem, \
                         ls='--', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)            
            ax[2].plot(np.arange(staMUA_att_mean.shape[0])/10, staMUA_att_mean, label='staMUA,att;%dHz'%stim_amp[stim_id])
            ax[2].fill_between(np.arange(staMUA_att_mean.shape[0])/10, staMUA_att_mean-staMUA_att_sem, staMUA_att_mean+staMUA_att_sem, \
                         ls='--', facecolor=clr[1], edgecolor=clr[1], alpha=0.2)            
       
    
    
        else:
            ax[0,stim_id].plot(freq[:plt_len], cohe_noatt_mean[:plt_len], label='cohe,noatt;%dHz'%stim_amp[stim_id])
            ax[0,stim_id].fill_between(freq[:plt_len], cohe_noatt_mean[:plt_len]-cohe_noatt_sem[stim_id], cohe_noatt_mean[:plt_len]+cohe_noatt_sem[stim_id], \
                         ls='--', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)            
            ax[0,stim_id].plot(freq[:plt_len], cohe_att_mean[:plt_len], label='cohe,att;%dHz'%stim_amp[stim_id])
            ax[0,stim_id].fill_between(freq[:plt_len], cohe_att_mean[:plt_len]-cohe_att_sem[stim_id], cohe_att_mean[:plt_len]+cohe_att_sem[stim_id], \
                         ls='--', facecolor=clr[1], edgecolor=clr[1], alpha=0.2)            

            ax[1,stim_id].plot(freq[:plt_len], staMUA_pw_noatt_mean[:plt_len], label='staMUA_pw,noatt;%dHz'%stim_amp[stim_id])
            ax[1,stim_id].fill_between(freq[:plt_len], staMUA_pw_noatt_mean[:plt_len]-staMUA_pw_noatt_sem[:plt_len], staMUA_pw_noatt_mean[:plt_len]+staMUA_pw_noatt_sem[:plt_len], \
                         ls='--', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)            
            ax[1,stim_id].plot(freq[:plt_len], staMUA_pw_att_mean[:plt_len], label='staMUA_pw,att;%dHz'%stim_amp[stim_id])
            ax[1,stim_id].fill_between(freq[:plt_len], staMUA_pw_att_mean[:plt_len]-staMUA_pw_att_sem[:plt_len], staMUA_pw_att_mean[:plt_len]+staMUA_pw_att_sem[:plt_len], \
                         ls='--', facecolor=clr[1], edgecolor=clr[1], alpha=0.2)            
            ax[1,stim_id].set_yscale('log',base=10)
            ax[1,stim_id].set_xscale('log',base=10)

                
        
            ax[2,stim_id].plot(np.arange(staMUA_noatt_mean.shape[0])/10, staMUA_noatt_mean, label='staMUA,noatt;%dHz'%stim_amp[stim_id])
            ax[2,stim_id].fill_between(np.arange(staMUA_noatt_mean.shape[0])/10, staMUA_noatt_mean-staMUA_noatt_sem, staMUA_noatt_mean+staMUA_noatt_sem, \
                         ls='--', facecolor=clr[0], edgecolor=clr[0], alpha=0.2)            
            ax[2,stim_id].plot(np.arange(staMUA_att_mean.shape[0])/10, staMUA_att_mean, label='staMUA,att;%dHz'%stim_amp[stim_id])
            ax[2,stim_id].fill_between(np.arange(staMUA_att_mean.shape[0])/10, staMUA_att_mean-staMUA_att_sem, staMUA_att_mean+staMUA_att_sem, \
                         ls='--', facecolor=clr[1], edgecolor=clr[1], alpha=0.2)            
        
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
    savetitle = savetitle+save_apd+'_%d'%(param_id)+'.png'
    if save_img: fig.savefig(save_dir + savetitle)
    plt.close()
