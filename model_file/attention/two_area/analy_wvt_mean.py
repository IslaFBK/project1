#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:27:30 2021

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
data_analy_file = 'data_anly_wvt'# data_anly_thr data_anly' #data_anly_fano data_anly_fano10hz data_anly_nc data_anly_fano10hz data_anly_fanorange
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
stim_amp = [200, 400, 800] #200*2**np.arange(n_StimAmp)

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
    
    wvt_resp_sens = []
    wvt_resp_asso = []
    
    for loop_num in range(param_id*repeat,(param_id+1)*repeat):
        
        try: 
            data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        except EOFError:
            print('EOFError: file: %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        
        wvt_resp_sens.append(data_anly.wvt.wvt_resp_sens)
        wvt_resp_asso.append(data_anly.wvt.wvt_resp_asso)
        
        # for stim_id in range(n_StimAmp):
        #     on_t_noatt[stim_id].append(np.concatenate(data_anly.onoff.stim_noatt[stim_id].on_t))
        #     #print(len(on_t_noatt[0]), '%d'%stim_id)
        #     off_t_noatt[stim_id].append(np.concatenate(data_anly.onoff.stim_noatt[stim_id].off_t))
        #     on_amp_noatt[stim_id].append(np.concatenate(data_anly.onoff.stim_noatt[stim_id].on_amp))
        #     off_amp_noatt[stim_id].append(np.concatenate(data_anly.onoff.stim_noatt[stim_id].off_amp))
            
        #     on_t_att[stim_id].append(np.concatenate(data_anly.onoff.stim_att[stim_id].on_t))
        #     off_t_att[stim_id].append(np.concatenate(data_anly.onoff.stim_att[stim_id].off_t))
        #     on_amp_att[stim_id].append(np.concatenate(data_anly.onoff.stim_att[stim_id].on_amp))
        #     off_amp_att[stim_id].append(np.concatenate(data_anly.onoff.stim_att[stim_id].off_amp))

    
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
    
    freq = data_anly.wvt.freq_sens      
    wvt_resp_sens = np.array(wvt_resp_sens) 
    wvt_resp_asso = np.array(wvt_resp_asso)
    wvt_resp_sens_m = np.nanmean(wvt_resp_sens,0)
    wvt_resp_asso_m = np.nanmean(wvt_resp_asso,0)
    
    '''sens'''
    #stim_dura = data.a1.param.stim1.stim_on[0,1] - data.a1.param.stim1.stim_on[0,0]
    fig, ax = plt.subplots(2,n_StimAmp, figsize=[12,6])
    ax2_hz = []
    for st in range(n_StimAmp):
        '''no att'''
       
        fig, ax[0,st], ax2 = fqa.plot_cwt(wvt_resp_sens_m[:,:,st], freq, base = 10, colorbar=True, fig=fig,ax=ax[0,st])
        ax[0,st].set_title('%.1fHz'%stim_amp[st])
        ax2_hz.append(ax2)
        
        '''att'''
        
        fig, ax[1,st], ax2 = fqa.plot_cwt(wvt_resp_sens_m[:,:,st+n_StimAmp], freq, base = 10, colorbar=True, fig=fig,ax=ax[1,st])
        ax[1,st].set_title('%.1fHz att'%stim_amp[st])
        ax2_hz.append(ax2)
                
    ax2_hz[0].set_ylabel('Hz')
    ax2_hz[1].set_ylabel('Hz')
    ax[1,1].set_xlabel('ms')
    
    title_ = title + '\n_wvt_sens_%d'%param_id
    fig.suptitle(title_)#+title)
    
    #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
    fig.savefig(save_dir + title_.replace('\n','')+'.png')
    plt.close()

#%%
    '''asso'''
      
    fig, ax = plt.subplots(2,n_StimAmp, figsize=[12,6])
      
    for st in range(n_StimAmp):
        '''no attention'''
        
        plt_freq_i = 27
        fig, ax[0,st], ax2 = fqa.plot_cwt(wvt_resp_asso_m[:,:,st][:plt_freq_i], freq[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[0,st])
        ax[0,st].set_title('%.1fHz'%stim_amp[st])
        ax2_hz.append(ax2)
    
        '''attention'''
        
        fig, ax[1,st], ax2 = fqa.plot_cwt(wvt_resp_asso_m[:,:,st+n_StimAmp][:plt_freq_i], freq[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[1,st])
        ax[1,st].set_title('%.1fHz att'%stim_amp[st])
        ax2_hz.append(ax2)
            
    ax2_hz[0].set_ylabel('Hz')
    ax2_hz[1].set_ylabel('Hz')
    ax[1,1].set_xlabel('ms')
    
    title_ = title + '\n_wvt_asso_%d'%param_id
    fig.suptitle(title_)#+title)
    
    #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
    fig.savefig(save_dir + title_.replace('\n','')+'.png')
    plt.close()

#%%
    '''wvt mean across time'''
    def find_fpeak(wvt_tmp, freq_tmp, minf = 40):
        return freq_tmp[freq_tmp>=minf][np.argmax(wvt_tmp[freq_tmp>=minf])]
    
    fig, ax = plt.subplots(2,2, figsize=[12,9])
    for st in range(n_StimAmp):
        
        freq_max = find_fpeak(wvt_resp_sens_m[:,:,st].mean(1)[::-1], freq[::-1], 40)        
        ax[0,0].plot(freq[::-1], wvt_resp_sens_m[:,:,st].mean(1)[::-1],ls='--',c=clr[st],label='stim:%.1fhz;no-att;pkf:%.2f'%(stim_amp[st],freq_max))        
        freq_max = find_fpeak(wvt_resp_sens_m[:,:,st+n_StimAmp].mean(1)[::-1], freq[::-1], 40)                
        ax[0,0].plot(freq[::-1], wvt_resp_sens_m[:,:,st+n_StimAmp].mean(1)[::-1],ls='-',c=clr[st],label='stim:%.1fhz;att;pkf:%.2f'%(stim_amp[st],freq_max))

        freq_max = find_fpeak(wvt_resp_asso_m[:,:,st].mean(1)[::-1], freq[::-1], 40)                
        ax[0,1].plot(freq[::-1], wvt_resp_asso_m[:,:,st].mean(1)[::-1],ls='--',c=clr[st],label='stim:%.1fhz;no-att;pkf:%.2f'%(stim_amp[st],freq_max))
        freq_max = find_fpeak(wvt_resp_asso_m[:,:,st+n_StimAmp].mean(1)[::-1], freq[::-1], 40)        
        ax[0,1].plot(freq[::-1], wvt_resp_asso_m[:,:,st+n_StimAmp].mean(1)[::-1],ls='-',c=clr[st],label='stim:%.1fhz;att;pkf:%.2f'%(stim_amp[st],freq_max))
        
        ax[0,0].set_title('sens'); ax[0,0].set_xlabel('Hz');  ax[0,0].set_ylabel('Amp'); 
        ax[0,1].set_title('asso'); ax[0,1].set_xlabel('Hz');  ax[0,1].set_ylabel('Amp'); 

        ax[1,0].plot(freq[::-1], wvt_resp_sens_m[:,:,st].mean(1)[::-1],ls='--',c=clr[st],label='stim:%.1fhz;no-att'%(stim_amp[st]))
        ax[1,0].plot(freq[::-1], wvt_resp_sens_m[:,:,st+n_StimAmp].mean(1)[::-1],ls='-',c=clr[st],label='stim:%.1fhz;att'%(stim_amp[st]))
        
        ax[1,1].plot(freq[::-1], wvt_resp_asso_m[:,:,st].mean(1)[::-1],ls='--',c=clr[st],label='stim:%.1fhz;no-att'%(stim_amp[st]))
        ax[1,1].plot(freq[::-1], wvt_resp_asso_m[:,:,st+n_StimAmp].mean(1)[::-1],ls='-',c=clr[st],label='stim:%.1fhz;att'%(stim_amp[st]))
        ax[1,0].set_xlabel('Hz');  ax[1,0].set_ylabel('Amp'); 
        ax[1,1].set_xlabel('Hz');  ax[1,1].set_ylabel('Amp'); 

    
        for axr in ax:
            for axc in axr:
                axc.legend(); #axc.legend()
        ax[1,0].set_xscale('log',base=10); ax[1,0].set_yscale('log',base=10)
        ax[1,1].set_xscale('log',base=10); ax[1,1].set_yscale('log',base=10)
        title_ = title + '\n_wvt_mean_%d'%param_id
        fig.suptitle(title_)#+title)
        
        #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
        fig.savefig(save_dir + title_.replace('\n','')+'.png')
        plt.close()

