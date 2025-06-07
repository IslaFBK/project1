#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:18:06 2021

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
import firing_rate_analysis as fra
import frequency_analysis as fqa
import fano_mean_match
#import find_change_pts
import HMM_py
import matlab.engine

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
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#loop_num = 0

good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly_onoff_hmm' #'data_anly' data_anly_temp
save_dir = 'hmm_results/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
#save_apd = ''

# onff_method = 'threshold'

# thre_spon = 5#4
# thre_stim = [20] #[12, 15, 30]

# fftplot = 1; getfano = 1
# get_nscorr = 1; get_nscorr_t = 1
# get_TunningCurve = 1; get_HzTemp = 1
# firing_rate_long = 1

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 0
else: get_ani = 0

save_analy_file = False
#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']


data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = [400] #200*2**np.arange(n_StimAmp)

if analy_type == 'fbrgbig4': # fbrg: feedback range
    title = '1irie%.2f_1e2e%.1f_pk2e1e%.2f'%(data.param.ie_r_i1, data.inter.param.w_e1_e2_mean/5, \
                                               data.inter.param.peak_p_e2_e1)
        
if analy_type == 'state': # fbrg: feedback range
    title = 'hz_2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                               data.inter.param.peak_p_e2_e1)
        
#%%

mua_loca = [0, 0]
mua_range = 5
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
#%%
def onoff_analysis(findonoff, spk_mat, mua_neuron, analy_dura_stim, n_StimAmp, n_perStimAmp, stim_amp, \
               title, save_apd, cross_validation, savefig):

    '''spon onoff'''
    '''sens area'''
    dt_samp = findonoff.mua_sampling_interval
    start = 5000; end = 20000
    analy_dura_spon = np.array([[start,end]])
    
    
    spon_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura_spon, cross_validation=False) # cross_validation must be False here since spontaneous activity only has one trial
    # if onff_method == 'threshold':
    #     spon_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold=thre_spon)
    # else:
    #     spon_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'comparing')
    
    R = mydata.mydata()
    R.spon = spon_onoff
    
    
    stim_num = 0
    end_plot_time = 3000
    n_ind, t = spk_mat[mua_neuron,analy_dura_spon[stim_num,0]*10:(analy_dura_spon[stim_num,0]+end_plot_time)*10].nonzero()
    
    
    
    fig,ax = plt.subplots(2,1, figsize=[15,6])
    # ax[0].plot(t/10, n_ind, '|')
    # ax[0].plot(show_stim_on)  
    # ax[1].plot(np.arange(mua_.shape[0]),mua_)
    #ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
    plt_mua_t = np.arange(len(spon_onoff.mua[stim_num]))*dt_samp
    
    ax[0].plot(plt_mua_t[:int(round(end_plot_time/dt_samp))], spon_onoff.mua[stim_num][:int(round(end_plot_time/dt_samp))]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
    for i in range(spon_onoff.cpts[stim_num].shape[0]):
        if spon_onoff.cpts[stim_num][i] >= end_plot_time : break
        #ax[0].plot([spon_onoff.cpts[stim_num][i],spon_onoff.cpts[stim_num][i]],[0,spon_onoff.mua[stim_num][:end_plot_time].max()], c=clr[1])
        ax[0].axvline(spon_onoff.cpts[stim_num][i], c=clr[1])
    ax[1].plot(plt_mua_t[:int(round(end_plot_time/dt_samp))], spon_onoff.onoff_bool[stim_num][:int(round(end_plot_time/dt_samp))]*len(mua_neuron))
    ax[1].plot(t/10, n_ind, '|')
    fig.suptitle(title + save_apd + 'on-off; plot')
    
    savetitle = title.replace('\n','')
    onofffile = savetitle+'_spon_t'+save_apd+'_%d'%(loop_num)+'.png'
    if savefig : 
        fig.savefig(save_dir + onofffile)
        plt.close()
    
    
    fig, ax = plt.subplots(1,4, figsize=[15,6])
    hr = ax[0].hist(np.concatenate(spon_onoff.on_t),bins=20, density=True)
    mu = np.concatenate(spon_onoff.on_t).mean()
    ax[0].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[0].set_title('on period; spon; mean:%.2f'%mu)
    hr = ax[1].hist(np.concatenate(spon_onoff.off_t),bins=20, density=True)
    mu = np.concatenate(spon_onoff.off_t).mean()
    ax[1].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[1].set_title('off period; spon; mean:%.2f'%mu)
    hr = ax[2].hist(np.concatenate(spon_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    mu = np.concatenate(spon_onoff.on_amp).mean()
    ax[2].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[2].set_title('on rate; spon; mean:%.2f'%mu)
    hr = ax[3].hist(np.concatenate(spon_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
    mu = np.concatenate(spon_onoff.off_amp).mean()
    ax[3].plot([mu,mu],[0,hr[0].max()*1.2])
    ax[3].set_title('off rate; spon; mean:%.2f'%mu)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    fig.suptitle(title + save_apd + '_spon_on-off dist')
    
    savetitle = title.replace('\n','')
    onofffile = savetitle+'_spon_dis'+save_apd+'_%d'%(loop_num)+'.png'
    if savefig :
        fig.savefig(save_dir + onofffile)
        plt.close()
    
    
    '''stim;'''
    R.stim_noatt = []
    R.stim_att = []
    
    for n in range(n_StimAmp):
        '''no att'''
        analy_dura = analy_dura_stim[n*n_perStimAmp:(n+1)*n_perStimAmp].copy()
        #analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
        analy_dura[:,0] += 500
        
        #analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
        # findonoff.MinThreshold = None # None # 1000
        # findonoff.MaxNumChanges = int(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2) #15
        # findonoff.smooth_window = None #52
        
        # start = 5000; end = 20000
        # analy_dura = np.array([[start,end]])
        
        #analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
        #analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
        #analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
        #stim_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura)
    
        # if onff_method == 'threshold':
        #     stim_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold=thre_stim[n])
        # else:
        #     stim_onoff = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'comparing')
        
        stim_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura, cross_validation=cross_validation) 
    
        
        R.stim_noatt.append(stim_onoff)
        
        stim_num = 0
        plt_dura = 2000 #ms
        n_ind, t = spk_mat[mua_neuron,analy_dura[stim_num,0]*10:(analy_dura[stim_num,0]+plt_dura)*10].nonzero()
        
        '''on-off plot'''
        fig, ax = plt.subplots(4,1, figsize=[15,12])
        '''no att'''
        plt_mua_t = np.arange(len(stim_onoff.mua[stim_num]))*dt_samp
        plt_dura_dt = int(round(plt_dura/dt_samp))
        ax[0].plot(plt_mua_t[:plt_dura_dt], stim_onoff.mua[stim_num][:plt_dura_dt]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
    
        #ax[0].plot(stim_onoff.mua[stim_num][:plt_dura]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
        for i in range(stim_onoff.cpts[stim_num].shape[0]):
            if stim_onoff.cpts[stim_num][i] >= plt_dura: break
            #ax[0].plot([stim_onoff.cpts[stim_num][i],stim_onoff.cpts[stim_num][i]],[0,80], c=clr[1])
            ax[0].axvline(stim_onoff.cpts[stim_num][i], c=clr[1])
        ax[0].set_title('stim; no att')
        ax[1].plot(plt_mua_t[:plt_dura_dt], stim_onoff.onoff_bool[stim_num][:plt_dura_dt]*len(mua_neuron))
        ax[1].plot(t/10, n_ind, '|')
        ax[0].xaxis.set_visible(False)
        ax[1].xaxis.set_visible(False) 
       
        # fig,ax = plt.subplots(2,1)
        # # ax[0].plot(t/10, n_ind, '|')
        # # ax[0].plot(show_stim_on)  
        # # ax[1].plot(np.arange(mua_.shape[0]),mua_)
        # #ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
        # ax[0].plot(stim_onoff.mua[stim_num])
        # for i in range(stim_onoff.cpts[stim_num].shape[0]):
        #     ax[0].plot([stim_onoff.cpts[stim_num][i],stim_onoff.cpts[stim_num][i]],[0,80], c=clr[1])
        # ax[1].plot(stim_onoff.onoff_bool[stim_num]*80)
        # ax[1].plot(t/10, n_ind, '|')
        
        # fig,ax = plt.subplots(2,2)
        # ax[0,0].hist(np.concatenate(stim_onoff.on_t),bins=20, density=True)
        # ax[0,1].hist(np.concatenate(stim_onoff.off_t),bins=20, density=True)
        # ax[1,0].hist(np.concatenate(stim_onoff.on_amp),bins=20, density=True)
        # ax[1,1].hist(np.concatenate(stim_onoff.off_amp),bins=20, density=True)
        # ax[0,0].set_yscale('log')
        # ax[0,1].set_yscale('log')
        
        '''att'''
        analy_dura = analy_dura_stim[(n+n_StimAmp)*n_perStimAmp:(n+n_StimAmp+1)*n_perStimAmp].copy()
        #analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
        #analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
        analy_dura[:,0] += 500
        
        # findonoff.MinThreshold = None # None # 1000
        # findonoff.MaxNumChanges = int(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2) #15
        # findonoff.smooth_window = None #52
        
        # start = 5000; end = 20000
        # analy_dura = np.array([[start,end]])
        
        #analy_dura = data.a1.param.stim1.stim_on[20:40].copy()
        #analy_dura[:,1] -= (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
        #analy_dura[:,0] += (np.round((analy_dura[:,1]-analy_dura[:,0])/2)).astype(int)
        #stim_onoff_att = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura)
        # if onff_method == 'threshold':
        #     stim_onoff_att = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold=thre_stim[n])
        # else:
        #     stim_onoff_att = findonoff.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura, method = 'comparing')    
        
        stim_onoff_att = findonoff.analyze(spk_mat[mua_neuron], analy_dura, cross_validation=cross_validation) 
        
        R.stim_att.append(stim_onoff_att)
        
        stim_num = 0
        plt_dura = 2000
        n_ind, t = spk_mat[mua_neuron,analy_dura[stim_num,0]*10:(analy_dura[stim_num,0]+plt_dura)*10].nonzero()
         
        '''att'''
        plt_mua_t = np.arange(len(stim_onoff_att.mua[stim_num]))*dt_samp
        plt_dura_dt = int(round(plt_dura/dt_samp))
        ax[2].plot(plt_mua_t[:plt_dura_dt], stim_onoff_att.mua[stim_num][:plt_dura_dt]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
    
        #ax[2].plot(stim_onoff_att.mua[stim_num][:plt_dura]/mua_neuron.shape[0]/(findonoff.mua_win/1000))
        for i in range(stim_onoff_att.cpts[stim_num].shape[0]):
            if stim_onoff_att.cpts[stim_num][i] >= plt_dura: break
            #ax[2].plot([stim_onoff_att.cpts[stim_num][i],stim_onoff_att.cpts[stim_num][i]],[0,80], c=clr[1])
            ax[2].axvline(stim_onoff_att.cpts[stim_num][i], c=clr[1])
    
        ax[2].set_title('stim; att')
        ax[3].plot(plt_mua_t[:plt_dura_dt], stim_onoff_att.onoff_bool[stim_num][:plt_dura_dt]*len(mua_neuron))
    
        #ax[3].plot(stim_onoff_att.onoff_bool[stim_num][:plt_dura]*80)
        ax[3].plot(t/10, n_ind, '|')
        ax[2].xaxis.set_visible(False)      
        fig.suptitle(title + save_apd + '_stim: %.1f hz'%stim_amp[n])
    
        savetitle = title.replace('\n','')
        onofffile = savetitle+'_stim%d_t'%n+save_apd+'_%d'%(loop_num)+'.png'
        if savefig :
            fig.savefig(save_dir + onofffile)
            plt.close()      
        
        
        # fig,ax = plt.subplots(2,1)
        # # ax[0].plot(t/10, n_ind, '|')
        # # ax[0].plot(show_stim_on)  
        # # ax[1].plot(np.arange(mua_.shape[0]),mua_)
        # #ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
        # ax[0].plot(stim_onoff_att.mua[stim_num])
        # for i in range(stim_onoff_att.cpts[stim_num].shape[0]):
        #     ax[0].plot([stim_onoff_att.cpts[stim_num][i],stim_onoff_att.cpts[stim_num][i]],[0,80], c=clr[1])
        # ax[1].plot(stim_onoff_att.onoff_bool[stim_num]*80)
        # ax[1].plot(t/10, n_ind, '|')
        fig,ax = plt.subplots(2,4, figsize=[15,6])
        hr = ax[0,0].hist(np.concatenate(stim_onoff.on_t),bins=20, density=True)
        mu = np.concatenate(stim_onoff.on_t).mean()
        ax[0,0].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0,0].set_title('on period; no att; mean:%.2f'%mu)
        hr = ax[0,1].hist(np.concatenate(stim_onoff.off_t),bins=20, density=True)
        mu = np.concatenate(stim_onoff.off_t).mean()
        ax[0,1].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0,1].set_title('off period; no att; mean:%.2f'%mu)
        hr = ax[0,2].hist(np.concatenate(stim_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        mu = (np.concatenate(stim_onoff.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
        ax[0,2].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0,2].set_title('on rate; no att; mean:%.2f'%mu)
        hr = ax[0,3].hist(np.concatenate(stim_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        mu = (np.concatenate(stim_onoff.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
        ax[0,3].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[0,3].set_title('off rate; no att; mean:%.2f'%mu)
        ax[0,0].set_yscale('log')
        ax[0,1].set_yscale('log')
        
        #fig,ax = plt.subplots(2,4)
        hr = ax[1,0].hist(np.concatenate(stim_onoff_att.on_t),bins=20, density=True)
        mu = np.concatenate(stim_onoff_att.on_t).mean()
        ax[1,0].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1,0].set_title('on period; att; mean:%.2f'%mu)
        hr = ax[1,1].hist(np.concatenate(stim_onoff_att.off_t),bins=20, density=True)
        mu = np.concatenate(stim_onoff_att.off_t).mean()
        ax[1,1].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1,1].set_title('off period; att; mean:%.2f'%mu)
        hr = ax[1,2].hist(np.concatenate(stim_onoff_att.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        mu = (np.concatenate(stim_onoff_att.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
        ax[1,2].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1,2].set_title('on rate; att; mean:%.2f'%mu)
        hr = ax[1,3].hist(np.concatenate(stim_onoff_att.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        mu = (np.concatenate(stim_onoff_att.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000)).mean()
        ax[1,3].plot([mu,mu],[0,hr[0].max()*1.2])
        ax[1,3].set_title('off rate; att; mean:%.2f'%mu)
        ax[1,0].set_yscale('log')
        ax[1,1].set_yscale('log')
        fig.suptitle(title + save_apd + '_stim: %.1f hz'%stim_amp[n])
    
        savetitle = title.replace('\n','')
        onofffile = savetitle+'_stim%d_dis'%n+save_apd+'%d'%(loop_num)+'.png'
        if savefig :
            fig.savefig(save_dir + onofffile)
            plt.close() 
        
        # ax[1,0].hist(np.concatenate(stim_onoff_att.on_t),bins=20, density=True)
        # ax[1,0].set_title('on period; att')
        # ax[1,1].hist(np.concatenate(stim_onoff_att.off_t),bins=20, density=True)
        # ax[1,1].set_title('off period; att')
        # ax[1,2].hist(np.concatenate(stim_onoff_att.on_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        # ax[1,2].set_title('on rate; att')
        # ax[1,3].hist(np.concatenate(stim_onoff_att.off_amp)/mua_neuron.shape[0]/(findonoff.mua_win/1000),bins=20, density=True)
        # ax[1,3].set_title('off rate; att')
        # ax[1,0].set_yscale('log')
        # ax[1,1].set_yscale('log')
        return R
#%%
'''onoff'''
#data_anly.onoff = mydata.mydata()

findonoff = HMM_py.HMM_onoff()
findonoff.mat_eng = matlab.engine.start_matlab('-nodisplay')
findonoff.mat_eng.addpath('/headnode1/shni2598/brian2/NeuroNet_brian/analysis/HMM_analysis')
#findonoff.mat_eng.quit()


data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10])

#spk_mat = data.a1.ge.spk_matrix
analy_dura_stim = data.a1.param.stim1.stim_on.copy()
cross_validation = True

data_anly.onoff_sens = onoff_analysis(findonoff, data.a1.ge.spk_matrix, mua_neuron, analy_dura_stim, n_StimAmp, n_perStimAmp, stim_amp, \
               title, save_apd='_sens_hmm', cross_validation=cross_validation, savefig=save_img)

data_anly.onoff_asso = onoff_analysis(findonoff, data.a2.ge.spk_matrix, mua_neuron, analy_dura_stim, n_StimAmp, n_perStimAmp, stim_amp, \
               title, save_apd='_asso_hmm', cross_validation=cross_validation, savefig=save_img)
    
findonoff.mat_eng.quit()
#%%
if get_ani:
    '''no att'''
    first_stim = 0 #1*n_perStimAmp -1; 
    last_stim = 0 #1*n_perStimAmp
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = start_time + 1000        #data.a1.param.stim1.stim_on[last_stim,1] + 300
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    stim_on_off = data.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    #stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    adpt = None
    #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_noatt_%d'%loop_num+'.mp4'
    
    # if loop_num%1 == 0:
    ani.save(moviefile)
    
    del ani

    '''att'''
    first_stim = n_perStimAmp #1*n_perStimAmp -1 + n_perStimAmp*n_StimAmp; 
    last_stim = n_perStimAmp #1*n_perStimAmp + n_perStimAmp*n_StimAmp
    start_time = data.a1.param.stim1.stim_on[first_stim,0] - 300
    end_time = start_time + 1000        #data.a1.param.stim1.stim_on[last_stim,1] + 300
    
    data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 15)
    data.a1.ge.get_centre_mass()
    data.a1.ge.overlap_centreandspike()
    
    data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 15)
    data.a2.ge.get_centre_mass()
    data.a2.ge.overlap_centreandspike()
    
    #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
    #frames = int(end_time - start_time)
    frames = data.a1.ge.spk_rate.spk_rate.shape[2]
    
    stim_on_off = data.a1.param.stim1.stim_on-start_time
    stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    #stim = [[[[31.5,31.5],[-0.5,63.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]]]
    stim = [[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    #stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]

    adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    #adpt = None
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_att_%d'%loop_num+'.mp4'
    
    # if loop_num%1 == 0:
    ani.save(moviefile)
    del ani
    
#%%
if save_analy_file:
    data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)
#%%







