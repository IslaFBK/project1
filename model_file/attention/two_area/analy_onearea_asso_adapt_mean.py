#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 21:45:14 2021

@author: shni2598
"""

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
fftplot = 1; getfano = 1; 
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1;

analy_type = 'asso'

save_dir = 'mean_results/'
data_analy_file = 'data_anly'#data_anly' #data_anly_fano data_anly_fano10hz data_anly_nc data_anly_fano10hz data_anly_fanorange
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']
   
data_anly = mydata.mydata()
data = mydata.mydata()

#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/fb/const_2ndstim/raw_data/'
datapath = 'raw_data/'

loop_num = 0

data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
data.load(datapath+'data%d.file'%(loop_num))

# stim_loc = [0,0]
# dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
# dist_bin = np.arange(0, 31.5*2**0.5, 2.5)

#%%
repeat = 5


#%%
'''fft'''
if fftplot:
    #freq_spon_a1 = data_anly.freq_spon_a1
    freq_spon_a2 = data_anly.freq_spon_a2
    freq_adapt_a2 = data_anly.freq_adapt_a2  #%
    
    #coef_spon_a1 = np.zeros([repeat, data_anly.coef_spon_a1.shape[0]])
    coef_spon_a2 = np.zeros([repeat, data_anly.coef_spon_a2.shape[0]])
    coef_adapt_a2 = np.zeros([repeat, data_anly.coef_adapt_a2.shape[0]]) #%
    
    #coef_spon_a1[:] = np.nan
    coef_spon_a2[:] = np.nan
    coef_adapt_a2[:] = np.nan  
    
'''spon rate'''
spon_rate = np.zeros([repeat, 1])
spon_rate[:] = np.nan
    
    
#%%
for i in range(0, 176):
    spon_rate[:] = np.nan
    # if get_TunningCurve:
    #     hz_loc_mean[:] = np.nan
    #     hz_loc_spon_mean[:] = np.nan
    # if get_HzTemp:
    #     hz_t_mean[:] = np.nan
    #     hz_loc_elec_mean[:] = np.nan
    if fftplot:
        #coef_spon_a1[:] = np.nan
        coef_spon_a2[:] = np.nan
        coef_adapt_a2[:] = np.nan #%
    # if getfano:
    #     #fano[:] = np.nan
    #     win_id = -1
    #     for win in data_anly.fano.win_lst:#[50,100,150]:
    #         win_id += 1
    #         fano[win_id][:] = np.nan
    # if get_nscorr:
    #     nc[:] = np.nan
    # if get_nscorr_t:
    #     nc_t[:] = np.nan

    for loop_num in range(i*repeat, (i+1)*repeat):
        try: 
            data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        except EOFError:
            print('EOFError: file: %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
                #divd -= 1
        # if get_TunningCurve:                    
        #     hz_loc_mean[:,:,loop_num%repeat] = np.nanmean(data_anly.hz_loc.reshape(n_StimAmp*2, n_perStimAmp, -1), 1)
        #     hz_loc_spon_mean[:,:,loop_num%repeat] = data_anly.hz_loc_spon
        # if get_HzTemp:
        #     hz_t_mean[:,:,loop_num%repeat] = np.nanmean(data_anly.hz_t.reshape(n_StimAmp*2, n_perStimAmp, -1), 1)
        #     #hz_loc_elec_mean[:,:,loop_num%repeat] = np.nanmean(data_anly.hz_loc_elec.reshape(n_StimAmp*2, n_perStimAmp, -1), 1)
        #     hz_loc_elec_mean[:,:,loop_num%repeat] = data_anly.hz_loc_elec_mean
        if fftplot:
            #coef_spon_a1[loop_num%repeat,:] = np.abs(data_anly.coef_spon_a1)
            coef_spon_a2[loop_num%repeat,:] = np.abs(data_anly.coef_spon_a2)
            coef_adapt_a2[loop_num%repeat,:] = np.abs(data_anly.coef_adapt_a2) #%
        
        # spon_rate[loop_num%repeat, 0] = data_anly.spon_rate1
        # spon_rate[loop_num%repeat, 1] = data_anly.spon_rate2
        # spon_rate[loop_num%repeat, 2] = data_anly.adapt_rate1
        # spon_rate[loop_num%repeat, 3] = data_anly.adapt_rate2
        spon_rate[loop_num%repeat, 0] = data_anly.spon_rate2

    found = False
    cannotfind = False
    file_id = loop_num
    while not found:
        if file_id < i*repeat:
            print('cannot find any data file for param: %d'%i)
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
    
    if analy_type == 'bkg': # fbrg: feedback range
        title = 'hz_bkge%.1f_bkgi%.1f'%(data.param.bkg_rate_e,data.param.bkg_rate_i)

    if analy_type == 'asso': # fbrg: feedback range
        title = 'hz_2erie%.2f_2irie%.2f_dgk%d'%(data.param.ie_r_e1, data.param.ie_r_i1, \
                                                   data.param.new_delta_gk)
        
    spon_rate_mean = np.nanmean(spon_rate, 0)
    title += '_2hz%.2f'%(spon_rate_mean[0])#, spon_rate_mean[2], spon_rate_mean[1], spon_rate_mean[3])
    #title += '_1hz%.2f_1adphz%.2f_2hz%.2f_2adphz%.2f'%(spon_rate_mean[0], spon_rate_mean[2], spon_rate_mean[1], spon_rate_mean[3])

    '''fft'''
    if fftplot:
        '''spon a1'''
        # fig, ax = fqa.plot_fft(freq_spon_a1, np.nanmean(coef_spon_a1, 0), \
        #                        freq_max1=20, freq_max2 = 200, fig=None, ax=None, show_theta=True, label='spon_a1')
    
        # fig.suptitle(title + ' spon a1')
        # fftfile = title.replace('\n','')+'_fft_a1_spon_%d'%(i)+'.png'
        # fig.savefig(save_dir + fftfile)
        # plt.close()
        '''spon a2'''
        fig, ax = fqa.plot_fft(freq_spon_a2, np.nanmean(coef_spon_a2, 0), \
                               freq_max1=20, freq_max2 = 200, fig=None, ax=None, show_theta=True, label='spon_a2')
    
        fig.suptitle(title + ' spon a2')
        fftfile = title.replace('\n','')+'_fft_a2_spon_%d'%(i)+'.png'
        fig.savefig(save_dir + fftfile)
        plt.close()
        '''adapt a2'''
        fig, ax = fqa.plot_fft(freq_adapt_a2, np.nanmean(coef_adapt_a2, 0), \
            freq_max1=20, freq_max2 = 200, fig=None, ax=None, show_theta=True, label='adapt_a2')
    
        fig.suptitle(title + ' adapt a2')
        fftfile = title.replace('\n','')+'_fft_a2_adapt_%d'%(i)+'.png'
        fig.savefig(save_dir + fftfile)
        plt.close()     
    
