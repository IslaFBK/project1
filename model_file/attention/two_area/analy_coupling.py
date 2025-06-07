#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 22:42:37 2021

@author: shni2598
"""

import brian2.numpy_ as np
from scipy import signal

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
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/test_longstim2/'+data_dir

sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly_fcorr' #'data_anly' data_anly_temp

fftplot = 1; get_wvt = 0
getfano = 1
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1
firing_rate_long = 1

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 1
else: get_ani = 0

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

if analy_type == 'state': # fbrg: feedback range
    title = 'hz_2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                               data.inter.param.peak_p_e2_e1)

#%%
n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = [400] #200*2**np.arange(n_StimAmp)

#%%
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp,0]

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
# fig, ax = plt.subplots(1,1)

# ax.plot(mua_1[data.a1.param.stim1.stim_on[0,0]:data.a1.param.stim1.stim_on[0,1]])

#%%
sigmod = lambda x, alpha : 1/(1+np.exp(-x/alpha)) - 0.5#+ 10
#%%
def plot_cwt_corr(d1, d2, xcorr, corr_high_bool, sigmod_alpha=2):
    
    win_len = int(round((xcorr.shape[0] + 1)/2))
    print(win_len)
    
    coef_1_, freq_1_ = fqa.mycwt(d1, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    coef_2_, freq_2_ = fqa.mycwt(d2, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    
    #coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
    
    # plt_freq_i = 27
    # plt_h = coef[:plt_freq_i].shape[0] -1
    
    fig, ax = plt.subplots(3,1, figsize=[10,8])
    # #fig, ax[3], ax2 = fqa.plot_cwt(coef_1[:plt_freq_i], freq_1[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[3])
    corr_im = ax[0].imshow(sigmod(xcorr[:,:-win_len+1], sigmod_alpha), aspect='auto')
    print('xcorr_shape:%d'%(xcorr[:,:-win_len+1].shape[1]))
    
    plt.colorbar(corr_im,ax=ax[0])
    if win_len%2 != 0:
        print('odd')
        fig, ax[1], ax_c1 = fqa.plot_cwt(coef_1_[:,int((win_len-1)/2):int((-win_len+1)/2)], freq_1_, base = 10, colorbar=True, fig=fig,ax=ax[1])
        print('coef_shape:%d'%(coef_1_[:,int((win_len-1)/2):int((-win_len+1)/2)].shape[1]))
        fig, ax[2], ax_c2 = fqa.plot_cwt(coef_2_[:,int((win_len-1)/2):int((-win_len+1)/2)], freq_2_, base = 10, colorbar=True, fig=fig,ax=ax[2])
    else:
        print('even')
        fig, ax[1], ax_c1 = fqa.plot_cwt(coef_1_[:,int((win_len-2)/2):int((-win_len)/2)], freq_1_, base = 10, colorbar=True, fig=fig,ax=ax[1])
        print('coef_shape:%d'%(coef_1_[:,int((win_len-2)/2):int((-win_len)/2)].shape[1]))
        fig, ax[2], ax_c2 = fqa.plot_cwt(coef_2_[:,int((win_len-2)/2):int((-win_len)/2)], freq_2_, base = 10, colorbar=True, fig=fig,ax=ax[2])
        
    
    ##xcorr_c.shape
    #corr_high_bool = xcorr[lag_0-ctr_period:lag_0+ctr_period+1].max(0)>= 2#  1.5
    
    #ax[0].plot(corr_high_bool*30, c=clr[1]) 
    ax[0].plot(corr_high_bool[:-win_len+1]*win_len, c=clr[1]) 
    print('corr_high_bool_shape:%d'%(corr_high_bool[:-win_len+1].shape[0]))
    return fig, ax

def get_xcorr_multisignal(d1, d2, passband, stim_dura, thre_percentile, mode='cov', return_xcorr=None):
    '''
    return_xcorr: 'None', integer, or 'all'; whether to return xcorr results
    '''
    coup = fqa.samefreq_coupling()
    ctr_period = int(round(1000/(passband.sum()/2)))
    coup.window = int(round(1000/(passband.sum()/2) * 2.5))
    print(coup.window)
    for dura_i, dura in enumerate(stim_dura): 
        start_t = dura[0] #data.a1.param.stim1.stim_on[st,0]
        end_t = dura[1] #data.a1.param.stim1.stim_on[st,1]
        d1_t = d1[start_t:end_t].copy()
        d2_t = d2[start_t:end_t].copy()
    
        #coup = fqa.samefreq_coupling()
        #passband = np.array([90,100])
        #ctr_period = int(round(1000/(passband.sum()/2)))
        #coup.window = int(round(1000/(passband.sum()/2) * 2.5))
        
        xcorr, phase_diff = coup.get_xcorr(d1_t, d2_t, passband, mode=mode)
        
        if dura_i == 0:
            xcorr_all = []
            phase_diff_all = []
        xcorr_all.append(xcorr.copy())
        phase_diff_all.append(phase_diff.copy())
    
    xcorr_all = np.hstack(xcorr_all)
    phase_diff_all = np.hstack(phase_diff_all)
    
    lag_0 = int(round((xcorr.shape[0]+1)/2-1))
    ##lag_range = 
    #thre_percentile = 0.7# threshold
    xcorr_max_all = xcorr_all.max(0)
    corr_thre = np.sort(xcorr_max_all)[int(thre_percentile*xcorr_max_all.shape[0])]; print(xcorr_max_all.shape[0])
    # corr_thre = 1.5 # threshold
    corr_high_bool = xcorr_all[lag_0-ctr_period:lag_0+ctr_period+1].max(0) >= corr_thre#  1.5

    hist_, bine = np.histogram(phase_diff_all[corr_high_bool], 20, range=(-np.pi,np.pi))
    # if dura_i == 0:
    #     hist_high_noatt = np.zeros(hist_.shape)
    #     hist_low_noatt = np.zeros(hist_.shape)
    hist_high_noatt = hist_.copy()
    hist_, bine = np.histogram(phase_diff_all[np.logical_not(corr_high_bool)], 20, range=(-np.pi,np.pi))
    hist_low_noatt = hist_.copy()  
    
    if return_xcorr is not None:
        if return_xcorr == 'all':
            
            return hist_high_noatt, hist_low_noatt, bine, corr_thre, np.array([xcorr_max_all,phase_diff_all]), xcorr_all, phase_diff_all, corr_high_bool
        else:
            return hist_high_noatt, hist_low_noatt, bine, corr_thre, np.array([xcorr_max_all,phase_diff_all]), xcorr_all[:,:return_xcorr], phase_diff_all[:return_xcorr], corr_high_bool[:return_xcorr]
    else:
        return hist_high_noatt, hist_low_noatt, bine, corr_thre, np.array([xcorr_max_all,phase_diff_all])
    


#%%
passband_list = np.array([[90,100],[120,130],[15,20]])
for st in range(n_StimAmp):
    '''no-att'''
    stim_dura = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
    for passband_i, passband in enumerate(passband_list):
        if passband_i == 0:
            hist_highcorr_noatt = []
            hist_lowcorr_noatt = []
            corr_thre_noatt = []
            xcorr_max_phase_diff_noatt = []
        # for dura_i, dura in enumerate(stim_dura):        
        #     # if dura_i == 0:
        #     #     hist_high_noatt = [];
        #     #     hist_low_noatt = [];
        #     #     #hist_att = []
                
        #     start_t = dura[0] #data.a1.param.stim1.stim_on[st,0]
        #     end_t = dura[1] #data.a1.param.stim1.stim_on[st,1]
        #     mua_1_t = mua_1[start_t:end_t].copy()
        #     mua_2_t = mua_2[start_t:end_t].copy()
        
        #     coup = fqa.samefreq_coupling()
        #     #passband = np.array([90,100])
        #     ctr_period = int(round(1000/(passband.sum()/2)))
        #     coup.window = int(round(1000/(passband.sum()/2) * 2.5))
            
        #     xcorr, phase_diff = coup.get_xcorr(mua_1_t, mua_2_t, np.array([90,100]), mode='cov')
            
        #     lag_0 = int(round((xcorr.shape[0]+1)/2-1))
        #     ##lag_range = 
        #     thre_percentile = 0.7# threshold
        #     xcorr_max = xcorr.max(0)
        #     corr_thre = np.sort(xcorr_max)[int(thre_percentile*xcorr_max.shape[0])]
        #     # corr_thre = 1.5 # threshold
        #     corr_high_bool = xcorr[lag_0-ctr_period:lag_0+ctr_period+1].max(0) >= corr_thre#  1.5
        
        #     hist_, bine = np.histogram(phase_diff[corr_high_bool], 20, range=(-np.pi,np.pi))
        #     if dura_i == 0:
        #         hist_high_noatt = np.zeros(hist_.shape)
        #         hist_low_noatt = np.zeros(hist_.shape)
        #     hist_high_noatt += hist_
        #     hist_, bine = np.histogram(phase_diff[np.logical_not(corr_high_bool)], 20, range=(-np.pi,np.pi))
        #     hist_low_noatt += hist_
            
        #     if dura_i == 0:
        #         plt_len = 10000 # ms
        #         sigmod_alpha = 2
        #         fig, ax = plot_cwt_corr(mua_1_t[:3*plt_len], mua_2_t[:3*plt_len], xcorr[:,:3*plt_len], corr_high_bool[:3*plt_len], lag_0, ctr_period,sigmod_alpha)
        #         ax[0].set_title('no-att;cross-cov;sig_alpha:%.1f;corr_thre:%.1f'%(sigmod_alpha,corr_thre));ax[1].set_title('sens');ax[2].set_title('asso')
        #         #fig.suptitle(title + '\nsens; stim:%.1f'%(stim_amp[st]))
        #         title_ = title + '\n_coupt_sti%.1f_noatt_%d'%(stim_amp[st],loop_num)
        #         fig.suptitle(title_)#+title)
                
        #         #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
        #         fig.savefig(title_.replace('\n','')+'.png')
        #         plt.close()  
        
        thre_percentile = 0.7
        plt_len = 15000
        hist_high_noatt, hist_low_noatt, bine, corr_thre, xcorr_max_phase_diff, xcorr_all, phase_diff_all, corr_high_bool = \
            get_xcorr_multisignal(mua_1, mua_2, passband, stim_dura, thre_percentile, mode='cov', return_xcorr=plt_len)

        #plt_len = 30000 # ms
        sigmod_alpha = 2
        fig, ax = plot_cwt_corr(mua_1[stim_dura[0,0]:stim_dura[0,0]+plt_len], mua_2[stim_dura[0,0]:stim_dura[0,0]+plt_len], xcorr_all, corr_high_bool, sigmod_alpha)
        ax[0].set_title('no-att;cross-cov;sig_alpha:%.1f;corr_thre:%.1f;band:[%.1f,%.1f]'%(sigmod_alpha,corr_thre,passband[0],passband[1]));ax[1].set_title('sens');ax[2].set_title('asso')
        ##fig.suptitle(title + '\nsens; stim:%.1f'%(stim_amp[st]))
        # title_ = title + '\n_coupt_sti%.1f_band%dto%d_noatt_%d'%(stim_amp[st],passband[0],passband[1],loop_num)
        # fig.suptitle(title_)#+title)
        
        # #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
        # fig.savefig(title_.replace('\n','')+'.png')
        # plt.close()  
        # del xcorr_all, phase_diff_all, corr_high_bool
        
        hist_highcorr_noatt.append(hist_high_noatt.copy())
        hist_lowcorr_noatt.append(hist_low_noatt.copy())
        corr_thre_noatt.append(corr_thre)
        xcorr_max_phase_diff_noatt.append(xcorr_max_phase_diff.copy())
        
    hist_highcorr_noatt = np.array(hist_highcorr_noatt)
    hist_lowcorr_noatt = np.array(hist_lowcorr_noatt)
    corr_thre_noatt = np.array(corr_thre_noatt)
        
    if st == 0:        
        data_anly.coup = mydata.mydata()
        data_anly.coup.noatt_corr_thresh = []
        data_anly.coup.hist_high_noatt = []
        data_anly.coup.hist_low_noatt = []
        data_anly.coup.xcorr_max_phase_diff_noatt = []
        
    data_anly.coup.noatt_corr_thresh.append(corr_thre_noatt.copy())
    data_anly.coup.hist_high_noatt.append(hist_highcorr_noatt.copy())
    data_anly.coup.hist_low_noatt.append(hist_lowcorr_noatt.copy())
    data_anly.coup.bine = bine
    data_anly.coup.passband_list = passband_list
    data_anly.coup.xcorr_max_phase_diff_noatt.append(xcorr_max_phase_diff_noatt) 
    
    '''att'''
    stim_dura = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
    for passband_i, passband in enumerate(passband_list):
        if passband_i == 0:
            hist_highcorr_att = []
            hist_lowcorr_att = []
            corr_thre_att = []    
            xcorr_max_phase_diff_att = []
    
    # for dura_i, dura in enumerate(stim_dura):        
    #     # if dura_i == 0:
    #     #     hist_att = []; 
    #     #     #hist_att = []
            
    #     start_t = dura[0] #data.a1.param.stim1.stim_on[st,0]
    #     end_t = dura[1] #data.a1.param.stim1.stim_on[st,1]
    #     mua_1_t = mua_1[start_t:end_t].copy()
    #     mua_2_t = mua_2[start_t:end_t].copy()
    
    #     coup = fqa.samefreq_coupling()
    #     passband = np.array([90,100])
    #     ctr_period = int(round(1000/(passband.sum()/2)))
    #     coup.window = int(round(1000/(passband.sum()/2) * 2.5))
        
    #     xcorr, phase_diff = coup.get_xcorr(mua_1_t, mua_2_t, np.array([90,100]), mode='cov')
        
    #     lag_0 = int(round((xcorr.shape[0]+1)/2-1))
    #     ##lag_range = 
    #     thre_percentile = 0.7# threshold
    #     xcorr_max = xcorr.max(0)
    #     corr_thre = np.sort(xcorr_max)[int(thre_percentile*xcorr_max.shape[0])]
    #     # corr_thre = 2
    #     corr_high_bool = xcorr[lag_0-ctr_period:lag_0+ctr_period+1].max(0) >= corr_thre#  1.5
    
    #     hist_, bine = np.histogram(phase_diff[corr_high_bool], 20, range=(-np.pi,np.pi))
    #     if dura_i == 0:
    #         hist_high_att = np.zeros(hist_.shape)
    #         hist_low_att = np.zeros(hist_.shape)
    #     hist_high_att += hist_
    #     hist_, bine = np.histogram(phase_diff[np.logical_not(corr_high_bool)], 20, range=(-np.pi,np.pi))
    #     hist_low_att += hist_           

        # if dura_i == 0:
        #     plt_len = 10000 # ms
        #     sigmod_alpha = 2
        #     fig, ax = plot_cwt_corr(mua_1_t[:plt_len], mua_2_t[:plt_len], xcorr[:,:plt_len], corr_high_bool[:plt_len], lag_0, ctr_period, sigmod_alpha)
        #     ax[0].set_title('att;cross-cov;sig_alpha:%.1f;corr_thre:%.1f'%(sigmod_alpha,corr_thre));ax[1].set_title('sens');ax[2].set_title('asso')           
        #     #fig.suptitle(title + '\nasso, stim:%.1f'%(stim_amp[st]))
        #     title_ = title + '\n_coupt_sti%.1f_att_%d'%(stim_amp[st],loop_num)
        #     fig.suptitle(title_)#+title)
            
        #     #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
        #     fig.savefig(title_.replace('\n','')+'.png')
        #     plt.close()  
    
    # if st == 0:        
    #     #data_anly.coup = mydata.mydata()
    #     data_anly.coup.att_corr_thresh = []
    #     data_anly.coup.hist_high_att = []
    #     data_anly.coup.hist_low_att = []
    
    # data_anly.coup.att_corr_thresh.append(corr_thre)
    # data_anly.coup.hist_high_att.append(hist_high_att.copy())
    # data_anly.coup.hist_low_att.append(hist_low_att.copy())
    

        thre_percentile = 0.7
        plt_len = 15000
        hist_high_att, hist_low_att, bine, corr_thre, xcorr_max_phase_diff, xcorr_all, phase_diff_all, corr_high_bool = \
            get_xcorr_multisignal(mua_1, mua_2, passband, stim_dura, thre_percentile, mode='cov', return_xcorr=plt_len)

        #plt_len = 30000 # ms
        sigmod_alpha = 2
        fig, ax = plot_cwt_corr(mua_1[stim_dura[0,0]:stim_dura[0,0]+plt_len], mua_2[stim_dura[0,0]:stim_dura[0,0]+plt_len], xcorr_all, corr_high_bool, sigmod_alpha)
        ax[0].set_title('att;cross-cov;sig_alpha:%.1f;corr_thre:%.1f;band:[%.1f,%.1f]'%(sigmod_alpha,corr_thre,passband[0],passband[1]));ax[1].set_title('sens');ax[2].set_title('asso')
        #fig.suptitle(title + '\nsens; stim:%.1f'%(stim_amp[st]))
        title_ = title + '\n_coupt_sti%.1f_band%dto%d_att_%d'%(stim_amp[st],passband[0],passband[1],loop_num)
        fig.suptitle(title_)#+title)
        
        #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
        fig.savefig(title_.replace('\n','')+'.png')
        plt.close()  
        del xcorr_all, phase_diff_all, corr_high_bool
                
        hist_highcorr_att.append(hist_high_att.copy())
        hist_lowcorr_att.append(hist_low_att.copy())
        corr_thre_att.append(corr_thre)
        xcorr_max_phase_diff_att.append(xcorr_max_phase_diff.copy())       
        
    hist_highcorr_att = np.array(hist_highcorr_att)
    hist_lowcorr_att = np.array(hist_lowcorr_att)
    corr_thre_att = np.array(corr_thre_att)
        
    if st == 0:        
        #data_anly.coup = mydata.mydata()
        data_anly.coup.att_corr_thresh = []
        data_anly.coup.hist_high_att = []
        data_anly.coup.hist_low_att = []
        data_anly.coup.xcorr_max_phase_diff_att = []
        
    data_anly.coup.att_corr_thresh.append(corr_thre_att.copy())
    data_anly.coup.hist_high_att.append(hist_highcorr_att.copy())
    data_anly.coup.hist_low_att.append(hist_lowcorr_att.copy())
    data_anly.coup.xcorr_max_phase_diff_att.append(xcorr_max_phase_diff_att) 


    for passband_i, passband in enumerate(passband_list):
        
        hist_highcorr_noatt = hist_highcorr_noatt/(hist_highcorr_noatt.sum(1).reshape(-1,1)); hist_highcorr_noatt /= bine[1]-bine[0]
        hist_lowcorr_noatt = hist_lowcorr_noatt/(hist_lowcorr_noatt.sum(1).reshape(-1,1)); hist_lowcorr_noatt /= bine[1]-bine[0]
    
        hist_highcorr_att = hist_highcorr_att/(hist_highcorr_att.sum(1).reshape(-1,1)); hist_highcorr_att /= bine[1]-bine[0]
        hist_lowcorr_att = hist_lowcorr_att/(hist_lowcorr_att.sum(1).reshape(-1,1)); hist_lowcorr_att /= bine[1]-bine[0]
        
        fig, ax = plt.subplots(2,2,figsize=[8,8])
        plt_bin = bine[:-1] + (bine[1]-bine[0])/2
        bar_size = bine[1]-bine[0]
        
        ax[0,0].bar(plt_bin, hist_highcorr_noatt[passband_i], bar_size, color=clr[0])
        ax[1,0].bar(plt_bin, hist_lowcorr_noatt[passband_i], bar_size, color=clr[0])
        ax[0,1].bar(plt_bin, hist_highcorr_att[passband_i], bar_size, color=clr[1])
        ax[1,1].bar(plt_bin, hist_lowcorr_att[passband_i], bar_size, color=clr[1])
        ax[0,0].set_title('high-coup;no-att')
        ax[1,0].set_title('low-coup;no-att')
        ax[0,1].set_title('high-coup;att')
        ax[1,1].set_title('low-coup;att')
        
        title_ = title + '\n_coup_sti%.1f_band%dto%d_%d'%(stim_amp[st],passband[0],passband[1],loop_num)
        fig.suptitle(title_)#+title)
        
        #if save_img: plt.savefig(title.replace('\n','')+'_tunecv'+'_%d'%loop_num+'.png')
        fig.savefig(title_.replace('\n','')+'.png')
        plt.close()

#%%
data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)

#%%
"""
#xcorr_c.shape
corr_high_bool = xcorr_c[20:41].max(0)>= 2#  1.5

ax[0].plot(corr_high_bool*30, c=clr[1])

#%%
plt.figure()
plt.hist(phase_diff_c[corr_high_bool],20)
#%%
fig, ax = plt.subplots(2,1, figsize=[6,8])
ax[0].hist(phase_diff_c[corr_high_bool],20,density=1)
ax[1].hist(phase_diff_c[np.logical_not(corr_high_bool)],20,density=1)

#%%
mode = 'cov'
d1 = mua_1_t_filt; d2 = mua_2_t_filt

window = 30; sample_interval = 1

dt = 1
sample_interval = int(np.round(sample_interval/dt))
window_step = int(np.round(window/dt))

hf_win = int(np.round(window_step/2))

start_time = 0 # int(np.round(start_time/dt))
end_time = d1.shape[0] # vint(np.round(end_time/dt))

xcorr_len = 2*window_step - 1
sample_t = np.arange(start_time, end_time-window_step+1, sample_interval)
xcorr = np.zeros([xcorr_len, sample_t.shape[0]])
phase_diff = np.zeros(sample_t.shape)

for ind, t in enumerate(sample_t):
    d1_seg = d1[t:t+window_step] - d1[t:t+window_step].mean() 
    d2_seg = d2[t:t+window_step] - d2[t:t+window_step].mean() 
    
    if mode = 'cov':
        xcorr_ = np.correlate(d1_seg, d2_seg, 'full')/window_step#/np.sqrt(np.sum(d1_seg**2) * np.sum(d2_seg**2))
    elif mode = 'corr':
        xcorr_ = np.correlate(d1_seg, d2_seg, 'full')/np.sqrt(np.sum(d1_seg**2) * np.sum(d2_seg**2))
        
    #xcorr_ d1[t:t+window_step]
    xcorr[:,ind] = xcorr_ #np.correlate(d1[t:t+window_step], d2[t:t+window_step], 'full')
    phase_diff[ind] = phase_dif[t+hf_win]
#return xcorr


#%%
Band = np.array([90,100])
Fs = 1000
#Wn = subBand[bandNum]/(Fs/2)
Wn = Band/(Fs/2)

filterOrder = 8
sos = signal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')

mua_1_t_filt = signal.sosfiltfilt(sos, mua_1_t, axis=-1)

mua_2_t_filt = signal.sosfiltfilt(sos, mua_2_t, axis=-1)

mua_1_t_filt_hil = signal.hilbert(mua_1_t_filt, axis=-1)
mua_2_t_filt_hil = signal.hilbert(mua_2_t_filt, axis=-1)
#%%
phase_dif = (np.angle(mua_1_t_filt_hil) - np.angle(mua_2_t_filt_hil) + np.pi)%(2*np.pi) - np.pi
#%%
fig, ax = plt.subplots(1,1)

ax.plot(phase_dif)
#%%
sigmod = lambda x : 1/(1+np.exp(-x/50)) - 0.5#+ 10
#%%

fig, ax = plt.subplots(1,1)
#corr_im = ax.imshow(np.log(xcorr[10:50]), aspect='auto')
corr_im = ax.imshow(sigmod(xcorr[10:50]), aspect='auto')

plt.colorbar(corr_im)

#%%
coef_1_, freq_1_ = fqa.mycwt(mua_1_t, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
#coef_2, freq_2 = fqa.mycwt(mua_2, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

#coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

# plt_freq_i = 27
# plt_h = coef[:plt_freq_i].shape[0] -1

# #fig, ax = plt.subplots(2,1, figsize=[10,6])
# #fig, ax[3], ax2 = fqa.plot_cwt(coef_1[:plt_freq_i], freq_1[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[3])
fqa.plot_cwt(coef_1_, freq_1_, base = 10, colorbar=True)#, fig=fig,ax=ax[3])
#%%
fig, ax = plt.subplots(1,1)

plt.plot(mua_1_t_filt)
plt.plot(np.abs(mua_1_t_filt_hil))

#%%
Band = np.array([40,100])
Fs = 1000
#Wn = subBand[bandNum]/(Fs/2)
Wn = Band/(Fs/2)

filterOrder = 8
sos = signal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')

mua_1_t_filt_2 = signal.sosfiltfilt(sos, mua_1_t, axis=-1)

mua_2_t_filt_2 = signal.sosfiltfilt(sos, mua_2_t, axis=-1)

mua_1_t_filt_hil_2 = signal.hilbert(mua_1_t_filt_2, axis=-1)
mua_2_t_filt_hil_2 = signal.hilbert(mua_2_t_filt_2, axis=-1)

#%%
fig, ax = plt.subplots(1,1)

plt.plot(mua_1_t_filt_2)
plt.plot(np.abs(mua_1_t_filt_hil_2))
#%%
mode = 'cov'
d1 = mua_1_t_filt; d2 = mua_2_t_filt

class samefreq_coupling:
    
    def __init__(self):    
        self.window = 30; 
        self.sample_interval = 1
        self.dt = 1 #ms
        
    def get_xcorr(self, data1, data2, band, mode = 'cov'):
        
        sample_interval = int(np.round(self.sample_interval/self.dt))
        window_step = int(np.round(self.window/self.dt))
        Fs = int(round(1/(self.dt*1e-3)))
        data1_filt, data2_filt, phase_diff_all = get_filt_phaseDiff(data1, data2, band, Fs = Fs, filterOrder = 8)
    
        hf_win = int(np.round(window_step/2))

        start_time = 0 # int(np.round(start_time/dt))
        end_time = data1_filt.shape[0] # vint(np.round(end_time/dt))

        xcorr_len = 2*window_step - 1
        sample_t = np.arange(start_time, end_time-window_step+1, sample_interval)
        xcorr = np.zeros([xcorr_len, sample_t.shape[0]])
        phase_diff = np.zeros(sample_t.shape)

        for ind, t in enumerate(sample_t):
            d1_seg = data1_filt[t:t+window_step] - data1_filt[t:t+window_step].mean() 
            d2_seg = data2_filt[t:t+window_step] - data2_filt[t:t+window_step].mean() 
            
            if mode == 'cov':
                xcorr_ = np.correlate(d1_seg, d2_seg, 'full')/window_step#/np.sqrt(np.sum(d1_seg**2) * np.sum(d2_seg**2))
            elif mode == 'corr':
                xcorr_ = np.correlate(d1_seg, d2_seg, 'full')/np.sqrt(np.sum(d1_seg**2) * np.sum(d2_seg**2))
            else: raise Exception("mode should be either 'cov' or 'corr'!")    
            #xcorr_ d1[t:t+window_step]
            xcorr[:,ind] = xcorr_ #np.correlate(d1[t:t+window_step], d2[t:t+window_step], 'full')
            phase_diff[ind] = phase_diff_all[t+hf_win]
        
        return xcorr, phase_diff
    
def get_filt_phaseDiff(data1, data2, band, Fs = 1000, filterOrder = 8):
    
    #Band = np.array([90,100])
    #Fs = 1000
    #Wn = subBand[bandNum]/(Fs/2)
    Wn = Band/(Fs/2)
    
    #filterOrder = 8
    sos = signal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')
    
    data1_filt = signal.sosfiltfilt(sos, data1, axis=-1)
    
    data2_filt = signal.sosfiltfilt(sos, data2, axis=-1)
    
    data1_filt_hil = signal.hilbert(data1_filt, axis=-1)
    data2_filt_hil = signal.hilbert(data2_filt, axis=-1)
    
    phase_diff = (np.angle(data1_filt_hil) - np.angle(data2_filt_hil) + np.pi)%(2*np.pi) - np.pi # phase1 - phase2
    
    return data1_filt, data2_filt, phase_diff
#%%
sigmod = lambda x, alpha : 1/(1+np.exp(-x/alpha)) - 0.5#+ 10


#%%
coup = fqa.samefreq_coupling()
xcorr_c, phase_diff_c = coup.get_xcorr(mua_1_t, mua_2_t, np.array([90,100]), mode='cov')

#mua_1_t
#%%
mua_1_t = mua_1[325230:425230].copy()
mua_2_t = mua_2[325230:425230].copy()
#%%
mua_1_t = mua_1[20000:120000].copy()
mua_2_t = mua_2[20000:120000].copy()


#%%

fig, ax = plt.subplots(1,1)
corr_im = ax.imshow(xcorr_c, aspect='auto')
#corr_im = ax.imshow(sigmod(xcorr[10:50]), aspect='auto')

plt.colorbar(corr_im)
#%%
fig, ax = plt.subplots(1,1)
#corr_im = ax.imshow(xcorr_c, aspect='auto')
corr_im = ax.imshow(sigmod(xcorr_c[10:50], 5/3), aspect='auto')

plt.colorbar(corr_im)




#%%
coef_1_, freq_1_ = fqa.mycwt(mua_1_t, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)
coef_2_, freq_2_ = fqa.mycwt(mua_2_t, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

#coef, freq = fqa.mycwt(mua, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

# plt_freq_i = 27
# plt_h = coef[:plt_freq_i].shape[0] -1

fig, ax = plt.subplots(3,1, figsize=[10,8])
# #fig, ax[3], ax2 = fqa.plot_cwt(coef_1[:plt_freq_i], freq_1[:plt_freq_i], base = 10, colorbar=True, fig=fig,ax=ax[3])
corr_im = ax[0].imshow(sigmod(xcorr_c[10:50], 5/3), aspect='auto')
plt.colorbar(corr_im,ax=ax[0])

fig, ax[1], ax_c1 = fqa.plot_cwt(coef_1_, freq_1_, base = 10, colorbar=True, fig=fig,ax=ax[1])
fig, ax[2], ax_c2 = fqa.plot_cwt(coef_2_, freq_2_, base = 10, colorbar=True, fig=fig,ax=ax[2])

#%%
#xcorr_c.shape
corr_high_bool = xcorr_c[20:41].max(0)>= 2#  1.5

ax[0].plot(corr_high_bool*30, c=clr[1])

#%%
plt.figure()
plt.hist(phase_diff_c[corr_high_bool],20)
#%%
fig, ax = plt.subplots(2,1, figsize=[6,8])
ax[0].hist(phase_diff_c[corr_high_bool],20,density=1)
ax[1].hist(phase_diff_c[np.logical_not(corr_high_bool)],20,density=1)

#%%
fig, ax = plt.subplots(1,1)

ax.plot(phase_dif)
"""
