#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:58:31 2021

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
import find_change_pts
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
data_dir = 'raw_data/'
analy_type = 'state'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/'+data_dir
#sys_argv = int(sys.argv[1])
#rep_ind*20 + ie_num
#loop_num = 0

good_dir = 'good/'
goodsize_dir = 'good_size/'

#savefile_name = 'data_anly' #'data_anly' data_anly_temp
save_dir = 'mean_results_hmm/'
data_analy_file = 'data_anly_onoff_hmm'#  data_anly_onoff_thres  data_anly_onoff_thres_samesens data_anly_onoff_hmm data_anly_thr data_anly' #data_anly_fano data_anly_fano10hz data_anly_nc data_anly_fano10hz data_anly_fanorange
# save_apd = '' #'_thr'
#save_apd_list=['_sens_thre', '_asso_thre'] # ['_sens_thre_samesens', '_asso_thre_samesens']
save_apd_list=['_sens_hmm', '_asso_hmm'] # ['_sens_thre_samesens', '_asso_thre_samesens']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fftplot = 1; getfano = 1; 
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1;

get_intvOn_var = 0
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

loop_num = 0 
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = [400] #200*2**np.arange(n_StimAmp)

mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)

mua_win = 10
        
#%%
#n_reliz = 20

#%%
n_param = 1
repeat = 20

for param_id in range(n_param):
    
    on_t_spon = [[],[]]
    off_t_spon = [[],[]]
    on_amp_spon = [[],[]]
    off_amp_spon = [[],[]]
    
    on_t_noatt = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]]
    off_t_noatt = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]] # [[] for _ in range(n_StimAmp)]
    on_amp_noatt = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]] # [[] for _ in range(n_StimAmp)]
    off_amp_noatt = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]] # [[] for _ in range(n_StimAmp)]
    if get_intvOn_var:
        off_t_var_noatt = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]]
    
    
    on_t_att = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]] # [[] for _ in range(n_StimAmp)]
    off_t_att = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]] # [[] for _ in range(n_StimAmp)]
    on_amp_att = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]] # [[] for _ in range(n_StimAmp)]
    off_amp_att = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]] # [[] for _ in range(n_StimAmp)]
    if get_intvOn_var:
        off_t_var_att = [[[] for _ in range(n_StimAmp)], [[] for _ in range(n_StimAmp)]]
    
    for loop_num in range(param_id*repeat,(param_id+1)*repeat):
        
        try: 
            data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        except EOFError:
            print('EOFError: file: %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        
        for aid, area in enumerate(['onoff_sens','onoff_asso']):
            on_t_spon[aid].append(np.concatenate(data_anly.__dict__[area].spon.on_t))
            off_t_spon[aid].append(np.concatenate(data_anly.__dict__[area].spon.off_t))
            on_amp_spon[aid].append(np.concatenate(data_anly.__dict__[area].spon.on_amp))
            off_amp_spon[aid].append(np.concatenate(data_anly.__dict__[area].spon.off_amp))
            
            for stim_id in range(n_StimAmp):
                on_t_noatt[aid][stim_id].append(np.concatenate(data_anly.__dict__[area].stim_noatt[stim_id].on_t))
                #print(len(on_t_noatt[0]), '%d'%stim_id)
                off_t_noatt[aid][stim_id].append(np.concatenate(data_anly.__dict__[area].stim_noatt[stim_id].off_t))
                on_amp_noatt[aid][stim_id].append(np.concatenate(data_anly.__dict__[area].stim_noatt[stim_id].on_amp))
                off_amp_noatt[aid][stim_id].append(np.concatenate(data_anly.__dict__[area].stim_noatt[stim_id].off_amp))
                
                
                on_t_att[aid][stim_id].append(np.concatenate(data_anly.__dict__[area].stim_att[stim_id].on_t))
                off_t_att[aid][stim_id].append(np.concatenate(data_anly.__dict__[area].stim_att[stim_id].off_t))
                on_amp_att[aid][stim_id].append(np.concatenate(data_anly.__dict__[area].stim_att[stim_id].on_amp))
                off_amp_att[aid][stim_id].append(np.concatenate(data_anly.__dict__[area].stim_att[stim_id].off_amp))
                
                if get_intvOn_var:
                    if aid == 1:
                        off_t_var_noatt[aid][stim_id].append(data_anly.__dict__[area].stim_noatt[stim_id].off_t_var)
                        off_t_var_att[aid][stim_id].append(data_anly.__dict__[area].stim_att[stim_id].off_t_var)
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

    for aid, save_apd in enumerate(save_apd_list):
        
        fig, ax = plt.subplots(1,4, figsize=[15,6])
        hr = ax[0].hist(np.concatenate(on_t_spon[aid]),bins=20, density=True)
        mu = np.concatenate(on_t_spon[aid]).mean()
        ax[0].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0].set_title('on period; spon; mean:%.2f'%mu)
        hr = ax[1].hist(np.concatenate(off_t_spon[aid]),bins=20, density=True)
        mu = np.concatenate(off_t_spon[aid]).mean()
        ax[1].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1].set_title('off period; spon; mean:%.2f'%mu)
        hr = ax[2].hist(np.concatenate(on_amp_spon[aid])/mua_neuron.shape[0]/(mua_win/1000),bins=20, density=True)
        mu = np.concatenate(on_amp_spon[aid]).mean()
        ax[2].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[2].set_title('on rate; spon; mean:%.2f'%mu)
        hr = ax[3].hist(np.concatenate(off_amp_spon[aid])/mua_neuron.shape[0]/(mua_win/1000),bins=20, density=True)
        mu = np.concatenate(off_amp_spon[aid]).mean()
        ax[3].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[3].set_title('off rate; spon; mean:%.2f'%mu)
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        fig.suptitle(title + save_apd + '_spon_on-off dist')
        
        savetitle = title.replace('\n','')
        onofffile = savetitle+'_spon_dis'+save_apd+'_%d'%(param_id)+'.png'
        fig.savefig(save_dir + onofffile)
        plt.close()
    
    
        for stim_id in range(n_StimAmp):
            
            fig, ax = plt.subplots(2,4, figsize=[15,6])
            hr = ax[0,0].hist(np.concatenate(on_t_noatt[aid][stim_id]),bins=20, density=True)
            mu = np.concatenate(on_t_noatt[aid][stim_id]).mean()
            ax[0,0].plot([mu,mu],[0,hr[0].max()*1.2])
            ax[0,0].set_title('on period; no att; mean:%.2f'%mu)
            hr = ax[0,1].hist(np.concatenate(off_t_noatt[aid][stim_id]),bins=20, density=True)
            mu = np.concatenate(off_t_noatt[aid][stim_id]).mean()
            ax[0,1].plot([mu,mu],[0,hr[0].max()*1.2])
            ax[0,1].set_title('off period; no att; mean:%.2f'%mu)
            hr = ax[0,2].hist(np.concatenate(on_amp_noatt[aid][stim_id])/mua_neuron.shape[0]/(mua_win/1000),bins=20, density=True)
            mu = (np.concatenate(on_amp_noatt[aid][stim_id])/mua_neuron.shape[0]/(mua_win/1000)).mean()
            ax[0,2].plot([mu,mu],[0,hr[0].max()*1.2])
            ax[0,2].set_title('on rate; no att; mean:%.2f'%mu)
            hr = ax[0,3].hist(np.concatenate(off_amp_noatt[aid][stim_id])/mua_neuron.shape[0]/(mua_win/1000),bins=20, density=True)
            mu = (np.concatenate(off_amp_noatt[aid][stim_id])/mua_neuron.shape[0]/(mua_win/1000)).mean()
            ax[0,3].plot([mu,mu],[0,hr[0].max()*1.2])
            ax[0,3].set_title('off rate; no att; mean:%.2f'%mu)
            ax[0,0].set_yscale('log')
            ax[0,1].set_yscale('log')
            
            #fig,ax = plt.subplots(2,4)
            hr = ax[1,0].hist(np.concatenate(on_t_att[aid][stim_id]),bins=20, density=True)
            mu = np.concatenate(on_t_att[aid][stim_id]).mean()
            ax[1,0].plot([mu,mu],[0,hr[0].max()*1.2])
            ax[1,0].set_title('on period; att; mean:%.2f'%mu)
            hr = ax[1,1].hist(np.concatenate(off_t_att[aid][stim_id]),bins=20, density=True)
            mu = np.concatenate(off_t_att[aid][stim_id]).mean()
            ax[1,1].plot([mu,mu],[0,hr[0].max()*1.2])
            ax[1,1].set_title('off period; att; mean:%.2f'%mu)
            hr = ax[1,2].hist(np.concatenate(on_amp_att[aid][stim_id])/mua_neuron.shape[0]/(mua_win/1000),bins=20, density=True)
            mu = (np.concatenate(on_amp_att[aid][stim_id])/mua_neuron.shape[0]/(mua_win/1000)).mean()
            ax[1,2].plot([mu,mu],[0,hr[0].max()*1.2])
            ax[1,2].set_title('on rate; att; mean:%.2f'%mu)
            hr = ax[1,3].hist(np.concatenate(off_amp_att[aid][stim_id])/mua_neuron.shape[0]/(mua_win/1000),bins=20, density=True)
            mu = (np.concatenate(off_amp_att[aid][stim_id])/mua_neuron.shape[0]/(mua_win/1000)).mean()
            ax[1,3].plot([mu,mu],[0,hr[0].max()*1.2])
            ax[1,3].set_title('off rate; att; mean:%.2f'%mu)
            ax[1,0].set_yscale('log')
            ax[1,1].set_yscale('log')
            fig.suptitle(title + save_apd + '_stim: %.1f hz'%stim_amp[stim_id])
        
            savetitle = title.replace('\n','')
            onofffile = savetitle+'_stim%d_dis'%stim_id+save_apd+'_%d'%(param_id)+'.png'
            fig.savefig(save_dir + onofffile)
            plt.close()  
            
            #fig, ax = plt.subplots(1,2, figsize=[10,5])
            #for n in range(n_StimAmp):
            if get_intvOn_var:
                if aid == 1: 
                    fig, ax = plt.subplots(1,2, figsize=[10,5])
                    ax[0].hist(np.concatenate(off_t_var_noatt[aid][stim_id]), bins=10, density=True)
                    ax[1].hist(np.concatenate(off_t_var_att[aid][stim_id]), bins=10, density=True)
                    ax[0].set_title('off period variance; no att')#; mean:%.2f'%mu)
                    ax[1].set_title('off period variance; att')#; mean:%.2f'%mu)
                    
                    fig.suptitle(title + save_apd + '_stim: %.1f hz'%stim_amp[stim_id])
                
                    savetitle = title.replace('\n','')
                    onofffile = savetitle+'_stim%d_offvar'%stim_id+save_apd+'_%d'%(param_id)+'.png'
                    fig.savefig(save_dir + onofffile)
                    plt.close()             





