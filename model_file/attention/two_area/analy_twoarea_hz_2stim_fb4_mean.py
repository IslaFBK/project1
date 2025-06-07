#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:43:17 2021

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

analy_type = 'wkfbrng'
    
save_dir = 'mean_results/'
data_analy_file = 'data_anly_fano10hz'#data_anly' #data_anly_fano data_anly_fano10hz data_anly_nc data_anly_fano10hz data_anly_fanorange
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

stim_loc = [0,0]
dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
# n_in_bin = [None]*(dist_bin.shape[0]-1)
# neuron = np.arange(data.a1.param.Ne)

# for i in range(len(dist_bin)-1):
#     n_in_bin[i] = neuron[(dist >= dist_bin[i]) & (dist < dist_bin[i+1])]
    
 



n_StimAmp = 3
n_perStimAmp = 50
stim_amp = 200*2**np.arange(n_StimAmp)

repeat = 20

'''tunning curve'''
if get_TunningCurve:
    hz_loc_mean = np.zeros([n_StimAmp*2, data_anly.hz_loc.shape[1], repeat]);
    hz_loc_spon_mean = np.zeros([2, data_anly.hz_loc_spon.shape[1], repeat]); 

    hz_loc_mean[:] = np.nan
    hz_loc_spon_mean[:] = np.nan
'''hz temp'''
#data_anly.hz_loc_elec_mean
if get_HzTemp:
    hz_t_mean = np.zeros([n_StimAmp*2, data_anly.hz_t.shape[1], repeat]); 
    hz_loc_elec_mean = np.zeros([n_StimAmp*2+2, data_anly.hz_loc_elec_mean.shape[1], repeat])
    hz_t_mean[:] = np.nan
    hz_loc_elec_mean[:] = np.nan
'''fft'''
if fftplot:
    freq_spon_a1 = data_anly.freq_spon_a1
    freq_spon_a2 = data_anly.freq_spon_a2
    freq_adapt_a2 = data_anly.freq_adapt_a2  #%
    
    coef_spon_a1 = np.zeros([repeat, data_anly.coef_spon_a1.shape[0]])
    coef_spon_a2 = np.zeros([repeat, data_anly.coef_spon_a2.shape[0]])
    coef_adapt_a2 = np.zeros([repeat, data_anly.coef_adapt_a2.shape[0]]) #%
    
    coef_spon_a1[:] = np.nan
    coef_spon_a2[:] = np.nan
    coef_adapt_a2[:] = np.nan  #%
'''fano'''
if getfano:
    #fano = np.zeros([data_anly.fano.fano_mean_sem.shape[0],data_anly.fano.fano_mean_sem.shape[1],repeat,data_anly.fano.fano_mean_sem[3]])
    fano = [None]*len(data_anly.fano.win_lst)
    win_id = -1
    for win in data_anly.fano.win_lst:#[50,100,150]:
        win_id += 1
        fano[win_id] = np.zeros([data_anly.fano.fano_mean_sem[win_id].shape[0], data_anly.fano.fano_mean_sem[win_id].shape[1], repeat]) #= np.zeros([n_StimAmp*2, fano_mean_noatt.shape[0], 2])
        fano[win_id][:] = np.nan
'''noise correlation'''
if get_nscorr:
    nc = np.zeros((repeat,) + data_anly.nscorr.nscorr.shape[:-1])
    nc[:] = np.nan
'''noise correlation t'''
if get_nscorr_t:
    ary_shape = data_anly.nscorr_t.nscorr_t.shape
    nc_t = np.zeros([repeat, ary_shape[0],ary_shape[2],ary_shape[3]])
    nc_t[:] = np.nan
'''spon rate'''
spon_rate = np.zeros([repeat, 4])
spon_rate[:] = np.nan









#fano[:] = np.nan
#%%
for i in range(200, 270):
    spon_rate[:] = np.nan
    if get_TunningCurve:
        hz_loc_mean[:] = np.nan
        hz_loc_spon_mean[:] = np.nan
    if get_HzTemp:
        hz_t_mean[:] = np.nan
        hz_loc_elec_mean[:] = np.nan
    if fftplot:
        coef_spon_a1[:] = np.nan
        coef_spon_a2[:] = np.nan
        coef_adapt_a2[:] = np.nan #%
    if getfano:
        #fano[:] = np.nan
        win_id = -1
        for win in data_anly.fano.win_lst:#[50,100,150]:
            win_id += 1
            fano[win_id][:] = np.nan
    if get_nscorr:
        nc[:] = np.nan
    if get_nscorr_t:
        nc_t[:] = np.nan

    for loop_num in range(i*repeat, (i+1)*repeat):
        try: 
            data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
                #divd -= 1
        if get_TunningCurve:                    
            hz_loc_mean[:,:,loop_num%repeat] = np.nanmean(data_anly.hz_loc.reshape(n_StimAmp*2, n_perStimAmp, -1), 1)
            hz_loc_spon_mean[:,:,loop_num%repeat] = data_anly.hz_loc_spon
        if get_HzTemp:
            hz_t_mean[:,:,loop_num%repeat] = np.nanmean(data_anly.hz_t.reshape(n_StimAmp*2, n_perStimAmp, -1), 1)
            #hz_loc_elec_mean[:,:,loop_num%repeat] = np.nanmean(data_anly.hz_loc_elec.reshape(n_StimAmp*2, n_perStimAmp, -1), 1)
            hz_loc_elec_mean[:,:,loop_num%repeat] = data_anly.hz_loc_elec_mean
        if fftplot:
            coef_spon_a1[loop_num%repeat,:] = np.abs(data_anly.coef_spon_a1)
            coef_spon_a2[loop_num%repeat,:] = np.abs(data_anly.coef_spon_a2)
            coef_adapt_a2[loop_num%repeat,:] = np.abs(data_anly.coef_adapt_a2) #%
        
        spon_rate[loop_num%repeat, 0] = data_anly.spon_rate1
        spon_rate[loop_num%repeat, 1] = data_anly.spon_rate2
        spon_rate[loop_num%repeat, 2] = data_anly.adapt_rate1
        spon_rate[loop_num%repeat, 3] = data_anly.adapt_rate2
        if getfano:
            #fano[:,:,loop_num%repeat,:] = data_anly.fano.fano_mean_sem[:,:,0,:]
            win_id = -1
            for win in data_anly.fano.win_lst:#[50,100,150]:
                win_id += 1
                fano[win_id][:,:,loop_num%repeat] = data_anly.fano.fano_mean_sem[win_id][:,:,0]
        if get_nscorr:
            nc[loop_num%repeat] = data_anly.nscorr.nscorr[:,:,:,0]
        if get_nscorr_t:
            nc_t[loop_num%repeat] = data_anly.nscorr_t.nscorr_t[:,0,:,:]
            
            
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
        
    # title = 'hz_1e2e%.2f_1e2i%.2f_2e1e%.2f_2e1i%.2f'%(data.inter.param.w_e1_e2_mean/5, data.inter.param.w_e1_i2_mean/5, \
    #                                                 data.inter.param.w_e2_e1_mean/5, data.inter.param.w_e2_i1_mean/5)
    if analy_type == 'wkfbrng': # weak feedback
        title = '2e1e%.2f_2e1i%.2f_2e1er%.1f_2e1ir%.1f'%(data.inter.param.w_e2_e1_mean/5, data.inter.param.w_e2_i1_mean/5,\
                                    data.inter.param.tau_p_d_e2_e1, data.inter.param.tau_p_d_e2_i1)    
        
    spon_rate_mean = np.nanmean(spon_rate, 0)
    title += '_1hz%.2f_1adphz%.2f_2hz%.2f_2adphz%.2f'%(spon_rate_mean[0], spon_rate_mean[2], spon_rate_mean[1], spon_rate_mean[3])
    
    if get_TunningCurve:
        hz_loc_mean_reliz = np.nanmean(hz_loc_mean, -1)
        hz_loc_sem_reliz = sem(hz_loc_mean, -1, nan_policy='omit')
        hz_loc_spon_mean_reliz = np.nanmean(hz_loc_spon_mean, -1)
        hz_loc_spon_sem_reliz = sem(hz_loc_spon_mean, -1, nan_policy='omit')
    if get_HzTemp:
        hz_t_mean_reliz = np.nanmean(hz_t_mean, -1)
        hz_t_sem_reliz = sem(hz_t_mean, -1, nan_policy='omit')
        hz_loc_elec_mean_reliz = np.nanmean(hz_loc_elec_mean, -1)
        hz_loc_elec_sem_reliz = sem(hz_loc_elec_mean, -1, nan_policy='omit')
    
    # n_StimAmp = 4
    # n_perStimAmp = 50
    # stim_amp = 200*2**np.arange(n_StimAmp)
    '''tunning curve'''
    if get_TunningCurve:
        dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]
        
        plt.figure(figsize=[8,6])
        for st in range(n_StimAmp):
            
            plt.errorbar(dist_bin_plot, hz_loc_mean_reliz[st], hz_loc_sem_reliz[st], fmt='--', marker='o', c=clr[st], label = 'stim_amp: %.1f Hz'%(stim_amp[st]))
            
            plt.errorbar(dist_bin_plot, hz_loc_mean_reliz[st+n_StimAmp], hz_loc_sem_reliz[st+n_StimAmp], fmt='-', marker='o', c=clr[st], label = 'att; stim_amp: %.1f Hz'%(stim_amp[st]))
    
        plt.errorbar(dist_bin_plot, hz_loc_spon_mean_reliz[0], hz_loc_spon_sem_reliz[0], fmt='--', marker='o', c=clr[st+1], label = 'spontaneous')
        plt.errorbar(dist_bin_plot, hz_loc_spon_mean_reliz[1], hz_loc_spon_sem_reliz[1], fmt='-', marker='o', c=clr[st+1], label = 'attention; spontaneous')
        
        plt.title(title + '\nSpike-count bin: %.1f ms\n'%(data.a1.param.stim1.stim_on[0,1]-data.a1.param.stim1.stim_on[0,0]))#+title)
        
        plt.xlim([dist_bin[0],dist_bin[-1]])
        plt.xlabel('distance')
        plt.ylabel('Hz')
        plt.legend()
        plt.savefig(save_dir + title.replace('\n','')+'_tunecv'+'_%d'%i+'.png')
        plt.close()
        
    '''firing rate - time'''    
    if get_HzTemp:
        
        dura_onoff = data.a1.param.stim1.stim_on[0,1] + 100 - data.a1.param.stim1.stim_on[0,0] + 100
        t_plot = np.arange(dura_onoff) - 100    
        mua_loca_1 = [0, 0]
        
        fig, ax, = plt.subplots(2,1, figsize=[8,10])    
        for st in range(n_StimAmp):
            ax[0].plot(t_plot, hz_t_mean_reliz[st], ls='--', c=clr[st], label = 'stim_amp: %.1f Hz;\nloc: [%.1f,%.1f]'%(stim_amp[st],mua_loca_1[0],mua_loca_1[1]))
            ax[0].fill_between(t_plot, hz_t_mean_reliz[st]-hz_t_sem_reliz[st], hz_t_mean_reliz[st]+hz_t_sem_reliz[st], \
                                     ls='--', facecolor=clr[st], edgecolor=clr[st], alpha=0.2)
                    #for st in range(n_amp_stim_att,n_amp_stim):
            ax[0].plot(t_plot, hz_t_mean_reliz[st+n_StimAmp], ls='-', c=clr[st], label = 'att; stim_amp: %.1f Hz;\nloc: [%.1f,%.1f]'%(stim_amp[st],mua_loca_1[0],mua_loca_1[1])) 
            ax[0].fill_between(t_plot, hz_t_mean_reliz[st+n_StimAmp]-hz_t_sem_reliz[st+n_StimAmp], hz_t_mean_reliz[st+n_StimAmp]+hz_t_sem_reliz[st+n_StimAmp], \
                                     ls='-', facecolor=clr[st], edgecolor=clr[st], alpha=0.2)
            
        ax[0].set_xlim([t_plot[0]+50, t_plot[-1]+100])
        ax[0].set_xlabel('ms')
        ax[0].set_ylabel('Hz')
        #title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
        #title = ''
        ax[0].set_title(title + '; senssory')
        ax[0].legend()
        
        mua_loca = [[0,0],[-10,0]]
        stim_amp_new = np.concatenate(([0],stim_amp))
        for lc in range(len(mua_loca)):
            ax[1].errorbar(np.arange(stim_amp_new.shape[0]), hz_loc_elec_mean_reliz[:n_StimAmp+1,lc], hz_loc_elec_sem_reliz[:n_StimAmp+1,lc], \
                         fmt='--', c=clr[lc], marker='o', label='no attention, loc: [%.1f,%.1f]'%(mua_loca[lc][0],mua_loca[lc][1]))
            ax[1].errorbar(np.arange(stim_amp_new.shape[0]), hz_loc_elec_mean_reliz[n_StimAmp+1:2*n_StimAmp+2,lc], hz_loc_elec_sem_reliz[n_StimAmp+1:2*n_StimAmp+2,lc], \
                         fmt='-', c=clr[lc], marker='o', label='attention, loc: [%.1f,%.1f]'%(mua_loca[lc][0],mua_loca[lc][1]))
        ax[1].set_ylabel('hz')
        
        ax_pct = ax[1].twinx()
        ax_pct.set_ylabel('rate increase percentage')
        for lc in range(len(mua_loca)):
            inc_percent = (hz_loc_elec_mean_reliz[n_StimAmp+1:2*n_StimAmp+2,lc] - hz_loc_elec_mean_reliz[:n_StimAmp+1,lc])/hz_loc_elec_mean_reliz[:n_StimAmp+1,lc]*100
            ax_pct.plot(np.arange(stim_amp_new.shape[0]), inc_percent, ls='-.', c=clr[lc], label='loc: [%.1f,%.1f]'%(mua_loca[lc][0],mua_loca[lc][1]))
        
        ax_pct.legend()
        ax[1].legend()
        ax[1].xaxis.set_ticks([st_ind for st_ind in range(len(stim_amp_new))])
        ax[1].xaxis.set_ticklabels([str(item) for item in stim_amp_new])     
        
        
        # for lc in range(len(mua_loca)):
        #     ax[1].errorbar(np.arange(stim_amp.shape[0]), hz_loc_elec_mean_reliz[:n_StimAmp,lc], hz_loc_elec_sem_reliz[:n_StimAmp,lc], \
        #                  fmt='--', c=clr[lc], marker='o', label='no attention, loc: [%.1f,%.1f]'%(mua_loca[lc][0],mua_loca[lc][1]))
        #     ax[1].errorbar(np.arange(stim_amp.shape[0]), hz_loc_elec_mean_reliz[n_StimAmp:2*n_StimAmp,lc], hz_loc_elec_sem_reliz[n_StimAmp:2*n_StimAmp,lc], \
        #                  fmt='-', c=clr[lc], marker='o', label='attention, loc: [%.1f,%.1f]'%(mua_loca[lc][0],mua_loca[lc][1]))
    
        # ax[1].legend()
        # ax[1].xaxis.set_ticks([st_ind for st_ind in range(len(stim_amp))])
        # ax[1].xaxis.set_ticklabels([str(item) for item in stim_amp])  
        plt.suptitle(title)# + '\nSpike-count bin: %.1f ms\n'%(data.a1.param.stim1.stim_on[0,1]-data.a1.param.stim1.stim_on[0,0]))#+title)
    
        #plt.savefig(save_dir+title.replace(':','')+'_temp'+'_%d'%ie_ind+'.png')
        fig.savefig(save_dir + title.replace('\n','')+'_temp'+'_%d'%i+'.png')
        plt.close()
    
    '''fft'''
    if fftplot:
        '''spon a1'''
        fig, ax = fqa.plot_fft(freq_spon_a1, np.nanmean(coef_spon_a1, 0), \
                               freq_max1=20, freq_max2 = 200, fig=None, ax=None, show_theta=True, label='spon_a1')
    
        fig.suptitle(title + ' spon a1')
        fftfile = title.replace('\n','')+'_fft_a1_spon_%d'%(i)+'.png'
        fig.savefig(save_dir + fftfile)
        plt.close()
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
    '''fano'''
    if getfano:
        bin_count_interval_hz = data_anly.fano.bin_count_interval_hz # 5
        win_id = -1
        for win in data_anly.fano.win_lst:
            win_id += 1
            fig, ax = plt.subplots(1,1, figsize=[8,6])
            fano_mean = np.nanmean(fano[win_id], 2)
            fano_sem = sem(fano[win_id],2,nan_policy='omit')
            for st in range(n_StimAmp):
                ax.errorbar(np.arange(fano_mean.shape[1])*10+(win/2), \
                            fano_mean[st,:], fano_sem[st,:], \
                            fmt='--', c=clr[st], marker='o', label='no att, stim_amp: %.1f Hz'%(stim_amp[st]))
                ax.errorbar(np.arange(fano_mean.shape[1])*10+(win/2),\
                            fano_mean[st+n_StimAmp,:], fano_sem[st+n_StimAmp,:],\
                            fmt='-', c=clr[st], marker='o', label='att, stim_amp: %.1f Hz'%(stim_amp[st]))
            ax.set_xlabel('ms')
            ax.set_ylabel('fano')
            plt.legend()
            title3 = title + '_win%.1f_bin%dhz\n_range%d'%(win, bin_count_interval_hz, data_anly.fano.neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
            fig.suptitle(title3)
            savetitle = title3.replace('\n','')
            fanofile = savetitle+'_%d'%(i)+'.png'
            fig.savefig(save_dir + fanofile)
            plt.close()
    '''noise correlation'''
    if get_nscorr:
        nc_mean = np.mean(nc, 0)
        nc_sem = sem(nc,0,nan_policy='omit')
        
        fig, ax = plt.subplots(1,1, figsize=[8,6])
        ax.errorbar(np.arange(len(stim_amp)), nc_mean[0,:,0], nc_sem[0,:,0], c=clr[0], fmt='--', marker='o', label='no-att;1-group')
        ax.errorbar(np.arange(len(stim_amp)), nc_mean[0,:,1], nc_sem[0,:,1], c=clr[1], fmt='-', marker='o', label='att;1-group')
        ax.errorbar(np.arange(len(stim_amp)), nc_mean[1,:,0], nc_sem[1,:,0], c=clr[0], fmt='--', marker='x', label='no-att;2-group')
        ax.errorbar(np.arange(len(stim_amp)), nc_mean[1,:,1], nc_sem[1,:,1], c=clr[1], fmt='-', marker='x', label='att;2-group')
    
        #ax.legend()
        ax.legend()
        ax.xaxis.set_ticks([i for i in range(len(stim_amp))])
        ax.xaxis.set_ticklabels([str(item) for item in stim_amp])     
        #title3 = title + '_win%.1f_bin%d\n_range%d'%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
        fig.suptitle(title)
        savetitle = title.replace('\n','')
        ncfile = savetitle+'_nc_%d'%(i)+'.png'
        fig.savefig(save_dir + ncfile)
        plt.close()
    
    '''noise correlation temporal'''
    if get_nscorr_t:
        nc_t_mean = np.mean(nc_t, 0)
        nc_t_sem = sem(nc_t,0,nan_policy='omit')
        
        fig, ax = plt.subplots(2,1, figsize=[8,6])
        #data_anly.nscorr_t.param.move_step = 20; data_anly.nscorr_t.param.t_bf = -50
        #nscorr_t.param
        sample_t = np.arange(nc_t_mean.shape[1])*data_anly.nscorr_t.param.move_step-data_anly.nscorr_t.param.t_bf
        for st in range(n_StimAmp):  
            ax[0].errorbar(sample_t, nc_t_mean[0, :, st], nc_t_sem[0, :, st], \
                           c=clr[st], fmt='--', marker='o', label='no-att;1-group;amp:%.1fHz'%stim_amp[st])
            ax[0].errorbar(sample_t, nc_t_mean[0, :, st+n_StimAmp], nc_t_sem[0, :, st+n_StimAmp], \
                           c=clr[st], fmt='-', marker='o', label='att;1-group;amp:%.1fHz'%stim_amp[st])
            ax[1].errorbar(sample_t, nc_t_mean[1, :, st], nc_t_sem[1, :, st], \
                           c=clr[st], fmt='--', marker='x', label='no-att;2-group;amp:%.1fHz'%stim_amp[st])
            ax[1].errorbar(sample_t, nc_t_mean[1, :, st+n_StimAmp], nc_t_sem[1, :, st+n_StimAmp], \
                           c=clr[st], fmt='-', marker='x', label='att;2-group;amp:%.1fHz'%stim_amp[st])

        ax[0].legend()
        ax[0].set_xlim([sample_t.min()-20,sample_t.max()+150])
        ax[1].legend()
        ax[1].set_xlim([sample_t.min()-20,sample_t.max()+150])
        fig.suptitle(title)
        savetitle = title.replace('\n','')
        ncfile = savetitle+'_nct_%d'%(i)+'.png'
        fig.savefig(save_dir + ncfile)
        plt.close()

        
#%%
# fig, ax = plt.subplots(1,1)
# #plt.errorbar(np.arange(3),np.arange(3), np.arange(3),fmt='--',marker='o',c='r')
# ax.plot(arange(400),label='aa\naa')
# ax.legend()
# ax.set_xlim([-20,550])
# #%%
# ax.xaxis.set_ticks([i for i in range(len(stim_amp))])
# ax.xaxis.set_ticklabels([str(item) for item in stim_amp])
# #ax.get_xticklabels()   