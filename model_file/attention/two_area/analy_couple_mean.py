#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 19:36:19 2021

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
save_dir = 'mean_results/'
data_analy_file = 'data_anly_fcorr'# data_anly_thr data_anly' #data_anly_fano data_anly_fano10hz data_anly_nc data_anly_fano10hz data_anly_fanorange
save_apd = '' #'_thr'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fftplot = 1; getfano = 1; 
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1;

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

# mua_win = 10
        
#%%
#n_reliz = 20

#%%
n_param = 1 #48
repeat = 20


for param_id in range(n_param):
    
    # wvt_resp_sens = []
    # wvt_resp_asso = []
    
    hist_highcorr_noatt = [] #[[] for _ in stim_amp]
    hist_lowcorr_noatt = [] #[[] for _ in stim_amp]
    corr_thre_noatt = [] #[[] for _ in stim_amp]
    
    hist_highcorr_att = [] #[[] for _ in stim_amp]
    hist_lowcorr_att = [] #[[] for _ in stim_amp]
    corr_thre_att = [] #[[] for _ in stim_amp]
    
    for loop_num in range(param_id*repeat,(param_id+1)*repeat):
        
        try: 
            data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        except EOFError:
            print('EOFError: file: %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        
        #wvt_resp_sens.append(data_anly.wvt.wvt_resp_sens)
        #wvt_resp_asso.append(data_anly.wvt.wvt_resp_asso)
        #data_anly.coup.passband_list = passband_list

        corr_thre_noatt.append(np.array(data_anly.coup.noatt_corr_thresh))
        hist_highcorr_noatt.append(np.array(data_anly.coup.hist_high_noatt))
        hist_lowcorr_noatt.append(np.array(data_anly.coup.hist_low_noatt)) 
        corr_thre_att.append(np.array(data_anly.coup.att_corr_thresh))
        hist_highcorr_att.append(np.array(data_anly.coup.hist_high_att))
        hist_lowcorr_att.append(np.array(data_anly.coup.hist_low_att))      

    bine = data_anly.coup.bine 
    passband_list = data_anly.coup.passband_list

    hist_highcorr_noatt = np.array(hist_highcorr_noatt) #[[] for _ in stim_amp]
    hist_lowcorr_noatt = np.array(hist_lowcorr_noatt) #[[] for _ in stim_amp]
    corr_thre_noatt = np.array(corr_thre_noatt) #[[] for _ in stim_amp]
    
    hist_highcorr_att = np.array(hist_highcorr_att) #[[] for _ in stim_amp]
    hist_lowcorr_att = np.array(hist_lowcorr_att) #[[] for _ in stim_amp]
    corr_thre_att = np.array(corr_thre_att)
    
    hist_highcorr_noatt = hist_highcorr_noatt.sum(0)
    hist_lowcorr_noatt = hist_lowcorr_noatt.sum(0)
    corr_thre_noatt = corr_thre_noatt.mean(0)
    hist_highcorr_att = hist_highcorr_att.sum(0)
    hist_lowcorr_att = hist_lowcorr_att.sum(0)
    corr_thre_att = corr_thre_att.mean(0)
    
    hist_highcorr_noatt = hist_highcorr_noatt/hist_highcorr_noatt.sum(-1,keepdims=True)/(bine[1]-bine[0])
    hist_lowcorr_noatt = hist_lowcorr_noatt/hist_lowcorr_noatt.sum(-1,keepdims=True)/(bine[1]-bine[0])
    hist_highcorr_att = hist_highcorr_att/hist_highcorr_att.sum(-1,keepdims=True)/(bine[1]-bine[0])
    hist_lowcorr_att = hist_lowcorr_att/hist_lowcorr_att.sum(-1,keepdims=True)/(bine[1]-bine[0])
    

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
 
    for st_i,st in enumerate(stim_amp):
        
        for passband_i, passband in enumerate(passband_list):
                
            fig, ax = plt.subplots(2,2,figsize=[8,8])
            plt_bin = bine[:-1] + (bine[1]-bine[0])/2
            bar_size = bine[1]-bine[0]
            
            ax[0,0].bar(plt_bin, hist_highcorr_noatt[st_i,passband_i,:], bar_size, color=clr[0])
            ax[1,0].bar(plt_bin, hist_lowcorr_noatt[st_i,passband_i,:], bar_size, color=clr[0])
            ax[0,1].bar(plt_bin, hist_highcorr_att[st_i,passband_i,:], bar_size, color=clr[1])
            ax[1,1].bar(plt_bin, hist_lowcorr_att[st_i,passband_i,:], bar_size, color=clr[1])
            ax[0,0].set_title('high-coup;no-att;corr_thre%.2f'%(corr_thre_noatt[st_i,passband_i]))
            
            ax[1,0].set_title('low-coup;no-att')
            ax[0,1].set_title('high-coup;att;corr_thre%.2f'%(corr_thre_att[st_i,passband_i]))
            ax[1,1].set_title('low-coup;att')
            
            
            title_ = title + '\n_coup_sti%.1f_band%dto%d_%d'%(st,passband[0],passband[1],param_id)
            fig.suptitle(title_)#+title)
            
            #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
            fig.savefig(save_dir + title_.replace('\n','')+'.png')
            plt.close()

        
        