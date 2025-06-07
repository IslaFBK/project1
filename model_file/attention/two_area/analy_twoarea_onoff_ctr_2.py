#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:42:18 2021

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
#import find_change_pts
import detect_onoff_ctr
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

savefile_name = 'data_anly_onoff_thres' #'data_anly' data_anly_temp data_anly_onoff_thres_samesens
#save_apd = '' 
save_apd_sens='_sens_thre'  # _sens_thre_samesens
save_apd_asso='_asso_thre' # _asso_thre_samesens
save_apd_alignedmua = '_thre'
#onoff_detect_method = 'threshold'

def rate2mua(r):
    r=np.array(r); 
    return r*0.01*80
#%%
thre_spon_sens = rate2mua(7)#4
thre_stim_sens = rate2mua([[25,30]]) #[12, 15, 30] rate2mua([[25,30]]) rate2mua([[30,30]])
thre_spon_asso = rate2mua(10)
thre_stim_asso = rate2mua([[20, 40]]) #[12, 15, 30] [[stim1_noatt,stim2_att],...]

# fftplot = 1; getfano = 1
# get_nscorr = 1; get_nscorr_t = 1
# get_TunningCurve = 1; get_HzTemp = 1
# firing_rate_long = 1

if loop_num%4 == 0: save_img = 1
else: save_img = 0

if loop_num%10 ==0: get_ani = 1
else: get_ani = 0

save_analy_file = True
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
def onoff_analysis(findonoff, spk_mat, mua_neuron, analy_dura_stim, n_StimAmp, n_perStimAmp, stim_amp, \
                   threshold_range, mua_loc, \
               title=None, save_apd=None, savefig=False):

    '''spon onoff'''
    
    
    dt_samp = findonoff.mua_sampling_interval
    start = 5000; end = 20000
    analy_dura_spon = np.array([[start,end]])
    
    
    #spon_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura_spon, cross_validation=False) # cross_validation must be False here since spontaneous activity only has one trial
    # findonoff.MinThreshold = None # None # 1000
    # findonoff.MaxNumChanges = int(round(((analy_dura_spon[0,1]-analy_dura_spon[0,0]))/1000*8*1.2)) #15
    # findonoff.smooth_window = None #52

    # if onoff_detect_method == 'threshold':
    #     spon_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura_spon, method = 'threshold', threshold=onoffThreshold_spon)
    # else:
    #     spon_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura_spon, method = 'comparing')
    spon_onoff = findonoff.analyze(spk_mat, mua_neuron, analy_dura_spon, \
                                     threshold_range = threshold_range, mua_loc = mua_loc)
    
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
    fig.suptitle(title + save_apd + '_spon_on-off; plot')
    
    savetitle = title.replace('\n','')
    onofffile = savetitle+'_spon_t'+save_apd+'_%d'%(loop_num)+'.png'
    if savefig : 
        fig.savefig(onofffile)
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
        fig.savefig(onofffile)
        plt.close()

    
    '''stim;'''
    #R = mydata.mydata()
    #dt_samp = findonoff.mua_sampling_interval
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
        
        #stim_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura, cross_validation=cross_validation) 
        # findonoff.MinThreshold = None # None # 1000
        # findonoff.MaxNumChanges = int(round(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2)) #15
        # print(analy_dura[0,1]-analy_dura[0,0])
        # findonoff.smooth_window = None #52
    
        # if onoff_detect_method == 'threshold':
        #     stim_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold=onoffThreshold_stim[n][0])
        # else:
        #     stim_onoff = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura, method = 'comparing')
    
        stim_onoff = findonoff.analyze(spk_mat, mua_neuron, analy_dura, \
                                     threshold_range = threshold_range, mua_loc = mua_loc)

        
        R.stim_noatt.append(stim_onoff)
        
        stim_num = 0
        plt_dura = 2000 #ms
        n_ind, t = spk_mat[mua_neuron,analy_dura[stim_num,0]*10:(analy_dura[stim_num,0]+plt_dura)*10].nonzero()
        
        '''on-off plot'''
        fig, ax = plt.subplots(4,1, figsize=[15,12])
        '''no att'''
        plt_mua_t = np.arange(len(stim_onoff.mua[stim_num]))*dt_samp
        plt_dura_dt = int(round(plt_dura/dt_samp))
        print(plt_dura_dt)
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
        
        #stim_onoff_att = findonoff.analyze(spk_mat[mua_neuron], analy_dura, cross_validation=cross_validation) 
        # findonoff.MinThreshold = None # None # 1000
        # findonoff.MaxNumChanges = int(round(((analy_dura[0,1]-analy_dura[0,0]))/1000*8*1.2)) #15
        # findonoff.smooth_window = None #52
    
        # if onoff_detect_method == 'threshold':
        #     stim_onoff_att = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura, method = 'threshold', threshold=onoffThreshold_stim[n][1])
        # else:
        #     stim_onoff_att = findonoff.analyze(spk_mat[mua_neuron], analy_dura=analy_dura, method = 'comparing')
        
        stim_onoff_att = findonoff.analyze(spk_mat, mua_neuron, analy_dura, \
                                     threshold_range = threshold_range, mua_loc = mua_loc)

        
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
            fig.savefig(onofffile)
            plt.close()      
        
        
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
        onofffile = savetitle+'_stim%d_dis'%n+save_apd+'_%d'%(loop_num)+'.png'
        if savefig :
            fig.savefig(onofffile)
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
# stim_num = 0
# end_plot_time = 3000
# n_ind, t = data.a1.ge.spk_matrix[mua_neuron,analy_dura[stim_num,0]*10:(analy_dura[stim_num,0]+end_plot_time)*10].nonzero()

#%%
'''onoff'''
#data_anly.onoff = mydata.mydata()

# findonoff = HMM_py.HMM_onoff()
# findonoff.mat_eng = matlab.engine.start_matlab('-nodisplay')
# findonoff.mat_eng.addpath('/headnode1/shni2598/brian2/NeuroNet_brian/analysis/HMM_analysis')
# #findonoff.mat_eng.quit()
#%%

mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
simu_time_tot = data.param.simutime#29000

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10], 'csc')
data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10], 'csc')

#spk_mat = data.a1.ge.spk_matrix
analy_dura_stim = data.a1.param.stim1.stim_on.copy()
#cross_validation = True

#findonoff_cpts = find_change_pts.MUA_findchangepts()
#%%
findonoff_cpts = detect_onoff_ctr.detect_onoff()
findonoff_cpts.mua_sampling_interval = 10
# data_anly_2 = findonoff_cpts.analyze(data.a1.ge.spk_matrix, mua_neuron, data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy(), \
#                                      threshold_range=9, mua_loc = np.array([31.5,31.5]))
# #%%

# plt.figure()
# plt.plot(data_anly_2.mua[0])
# plt.plot(data_anly_2.onoff_bool[0]*50)

# #%%
# analy_dura = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
# #%%
# stim_num = 0
# plt_dura = 2000
# n_ind, t = spk_mat[mua_neuron,analy_dura[stim_num,0]*10:(analy_dura[stim_num,0]+plt_dura)*10].nonzero()
# #%%
# dt_samp = 10
# plt_mua_t = np.arange(len(data_anly_2.mua[stim_num]))*dt_samp
# plt_dura_dt = int(round(plt_dura/dt_samp))
        
# plt.figure()
# plt.plot(plt_mua_t[:plt_dura_dt], data_anly_2.onoff_bool[0][:plt_dura_dt]*80)
# plt.plot(t/10, n_ind, '|')



# #%%
# first_stim = 0 #1*n_perStimAmp -1; 
# last_stim = 0 #1*n_perStimAmp
# start_time = data.a1.param.stim1.stim_on[first_stim,0] + 1000
# end_time = start_time + 1000        #data.a1.param.stim1.stim_on[last_stim,1] + 300
# # #%%
# # data.a1.ge.get_spike_rate(start_time=data.a1.param.stim1.stim_on[0,0], end_time=data.a1.param.stim1.stim_on[0,1], \
# #                           sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
# # data.a1.ge.get_centre_mass()

# data.a1.ge.overlap_centreandspike()

# data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)
# data.a2.ge.get_centre_mass()
# data.a2.ge.overlap_centreandspike()

# #data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
# #frames = int(end_time - start_time)
# frames = data.a1.ge.spk_rate.spk_rate.shape[2]

# stim_on_off = data.a1.param.stim1.stim_on-start_time
# stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
# stim_on_off = np.array([0,1000])
# stim = [[[[31.5,31.5]], [stim_on_off], [[9]*stim_on_off.shape[0]]],None]
# #stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]

# adpt = None
# #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
# #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
# ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
#                                         frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
# # savetitle = title.replace('\n','')

# # moviefile = savetitle+'_noatt_%d'%loop_num+'.mp4'

# # # if loop_num%1 == 0:
# # ani.save(moviefile)


#%%
# data_anly.onoff_sens = onoff_analysis(findonoff_cpts, data.a1.ge.spk_matrix, mua_neuron, analy_dura_stim, n_StimAmp, n_perStimAmp, stim_amp, \
#                     onoff_detect_method='threshold', onoffThreshold_spon=thre_spon_sens, onoffThreshold_stim=thre_stim_sens, \
#                     title=title, save_apd=save_apd_sens, savefig=save_img)

#%%
data_anly.onoff_sens = onoff_analysis(findonoff_cpts, data.a1.ge.spk_matrix, mua_neuron, analy_dura_stim, n_StimAmp, n_perStimAmp, stim_amp, \
                   threshold_range=9, mua_loc = np.array([31.5,31.5]), \
               title=title, save_apd=save_apd_sens, savefig=save_img)  
#%%

# stim_onoff_att = findonoff_cpts.analyze(data.a1.ge.spk_matrix[mua_neuron], analy_dura=analy_dura_stim[0:2], method = 'threshold', threshold=20)

# #%%
# plt.figure()
# plt.plot(stim_onoff_att.mua[0])
# plt.plot(stim_onoff_att.onoff_bool[0]*80)

#%%
# data_anly.onoff_asso = onoff_analysis(findonoff_cpts, data.a2.ge.spk_matrix, mua_neuron, analy_dura_stim, n_StimAmp, n_perStimAmp, stim_amp, \
#                     onoff_detect_method='threshold', onoffThreshold_spon=thre_spon_asso, onoffThreshold_stim=thre_stim_asso, \
#                     title=title, save_apd=save_apd_asso, savefig=save_img)
    
#%%
data_anly.onoff_asso = onoff_analysis(findonoff_cpts, data.a2.ge.spk_matrix, mua_neuron, analy_dura_stim, n_StimAmp, n_perStimAmp, stim_amp, \
                   threshold_range=9, mua_loc = np.array([31.5,31.5]), \
               title=title, save_apd=save_apd_asso, savefig=save_img) 

#%%
def get_offduration_var(data):
    for st in range(len(data)):
        
        off_t_var = np.zeros(len(data[st].off_t))
    
        for trial in range(len(data[st].off_t)):
            off_t_var[trial] = data[st].off_t[trial].var()
        
        data[st].off_t_var = off_t_var
    
    return data

#%%
data_anly.onoff_asso.stim_noatt = get_offduration_var(data_anly.onoff_asso.stim_noatt)
data_anly.onoff_asso.stim_att = get_offduration_var(data_anly.onoff_asso.stim_att)
#%%
for n in range(n_StimAmp):

    fig, ax = plt.subplots(1,2, figsize=[10,5])
    ax[0].hist(data_anly.onoff_asso.stim_noatt[n].off_t_var, bins=10, density=True)
    ax[1].hist(data_anly.onoff_asso.stim_att[n].off_t_var, bins=10, density=True)
    ax[0].set_title('off period variance; no att')#; mean:%.2f'%mu)
    ax[1].set_title('off period variance; att')#; mean:%.2f'%mu)
    
    fig.suptitle(title + save_apd_asso + '_stim: %.1f hz'%stim_amp[n])

    savetitle = title.replace('\n','')
    onofffile = savetitle+'_stim%d_offvar'%n+save_apd_asso+'_%d'%(loop_num)+'.png'
    if save_img :
        fig.savefig(onofffile)
        plt.close() 
        
#%%
def transition_aligned_rate(area_transition, area_rate):
    #mua_len = 9491
    #cpts
    mean_mua_aligned_on2off = []
    mean_mua_aligned_off2on = []
    for n in range(n_StimAmp):
        hz_mean_on2off = np.zeros(201);
        hz_mean_off2on = np.zeros(201);
        num_on = 0
        num_off = 0
        for onset_t, offset_t, cpts, mua in zip(area_transition[n].onset_t, \
                                      area_transition[n].offset_t, \
                                 area_transition[n].cpts,\
                                area_rate[n].mua):
            mua_len = mua.shape[0]
            for ti, t in enumerate(onset_t):
                if t < 100 or t + 100 > mua_len - 1:
                    continue
                if np.sum((cpts >= t-100) & (cpts <= t+100)) > 1:
                    #print('skip%d'%t)
                    continue
                
                hz_mean_on2off += mua[t-100:t+100+1]
                num_on += 1
                
            for ti, t in enumerate(offset_t):
                if t < 100 or t + 100 > mua_len - 1:
                    continue
                if np.sum((cpts >= t-100) & (cpts <= t+100)) > 1:
                    #print('skip%d'%t)
                    continue
                
                hz_mean_off2on += mua[t-100:t+100+1]
                num_off += 1
        mean_mua_aligned_on2off.append(hz_mean_on2off/num_on)
        mean_mua_aligned_off2on.append(hz_mean_off2on/num_off)
        
    return mean_mua_aligned_on2off, mean_mua_aligned_off2on
#%%
mean_assomua_aligned_senson2off_noatt, mean_assomua_aligned_sensoff2on_noatt = \
    transition_aligned_rate(data_anly.onoff_sens.stim_noatt, data_anly.onoff_asso.stim_noatt)
    
mean_assomua_aligned_senson2off_att, mean_assomua_aligned_sensoff2on_att = \
    transition_aligned_rate(data_anly.onoff_sens.stim_att, data_anly.onoff_asso.stim_att)
#%%
mean_sensmua_aligned_assoon2off_noatt, mean_sensmua_aligned_assooff2on_noatt = \
    transition_aligned_rate(data_anly.onoff_asso.stim_noatt, data_anly.onoff_sens.stim_noatt)
    
mean_sensmua_aligned_assoon2off_att, mean_sensmua_aligned_assooff2on_att = \
    transition_aligned_rate(data_anly.onoff_asso.stim_att, data_anly.onoff_sens.stim_att)
#%%
data_anly.mean_assomua_aligned_senson2off_noatt = mean_assomua_aligned_senson2off_noatt
data_anly.mean_assomua_aligned_sensoff2on_noatt = mean_assomua_aligned_sensoff2on_noatt
data_anly.mean_assomua_aligned_senson2off_att = mean_assomua_aligned_sensoff2on_att
data_anly.mean_sensmua_aligned_assoon2off_noatt = mean_sensmua_aligned_assooff2on_noatt
data_anly.mean_sensmua_aligned_assoon2off_att = mean_sensmua_aligned_assooff2on_att
#hz_asso_mean /= num
#%%

# plt.figure()
# plt.plot(hz_mean_on2off/num_on)  
# plt.plot(hz_mean_off2on/num_off)  
# plt.axvline(101)
#%% 
for n in range(n_StimAmp):
    fig, ax = plt.subplots(2,2, figsize=[12,10])
    
    plt_t = np.arange(201)-100
    #plt.figure()
    ax[0,0].plot(plt_t, mean_assomua_aligned_senson2off_noatt[n], label = 'asso mua; sens on2off; noatt')  
    ax[0,0].plot(plt_t, mean_assomua_aligned_sensoff2on_noatt[n], label = 'asso mua; sens off2on; noatt')  
    ax[0,0].axvline(0)
    
    #plt.figure()
    ax[0,1].plot(plt_t, mean_assomua_aligned_senson2off_att[n], label = 'asso mua; sens on2off; att')  
    ax[0,1].plot(plt_t, mean_assomua_aligned_sensoff2on_att[n], label = 'asso mua; sens off2on; att')  
    ax[0,1].axvline(0)
    
    #plt.figure()
    ax[1,0].plot(plt_t, mean_sensmua_aligned_assoon2off_noatt[n], label = 'sens mua; asso on2off')  
    ax[1,0].plot(plt_t, mean_sensmua_aligned_assooff2on_noatt[n], label = 'sens mua; asso off2on')  
    ax[1,0].axvline(0)
    
    #plt.figure()
    ax[1,1].plot(plt_t, mean_sensmua_aligned_assoon2off_att[n], label = 'sens mua; asso on2off')  
    ax[1,1].plot(plt_t, mean_sensmua_aligned_assooff2on_att[n], label = 'sens mua; asso off2on')  
    ax[1,1].axvline(0)
    
    for axy in ax:
        for axx in axy:
            axx.legend()
            
    fig.suptitle(title + save_apd_alignedmua + '_stim: %.1f hz'%stim_amp[n])
    
    savetitle = title.replace('\n','')
    onofffile = savetitle+'_stim%d_alignmua'%n+save_apd_alignedmua+'_%d'%(loop_num)+'.png'
    if save_img :
        fig.savefig(onofffile)
        plt.close() 

#%%

# stim_onoff_att = findonoff_cpts.analyze(data.a2.ge.spk_matrix[mua_neuron], analy_dura=analy_dura_stim[0+20:2+20], method = 'threshold', threshold=24)

# #%%
# plt.figure()
# plt.plot(stim_onoff_att.mua[0])
# plt.plot(stim_onoff_att.onoff_bool[0]*80)
# # for i in range(stim_onoff_att.cpts[0].shape[0]):
# #     plt.axvline(stim_onoff_att.cpts[0][i], c=clr[1])

# #%%
# fig,ax = plt.subplots(2,1, figsize=[15,6])
# # ax[0].plot(t/10, n_ind, '|')
# # ax[0].plot(show_stim_on)  
# # ax[1].plot(np.arange(mua_.shape[0]),mua_)
# #ax[0].plot(np.array(smt_mua_mat[stim_on_new[20,0]:stim_on_new[20,1]]))
# ax[0].plot(spon_onoff.mua[stim_num][:end_plot_time])
# for i in range(spon_onoff.cpts[stim_num].shape[0]):
#     if spon_onoff.cpts[stim_num][i] >= end_plot_time : break
#     ax[0].plot([spon_onoff.cpts[stim_num][i],spon_onoff.cpts[stim_num][i]],[0,spon_onoff.mua[stim_num][:end_plot_time].max()], c=clr[1])
# ax[1].plot(spon_onoff.onoff_bool[stim_num][:end_plot_time]*80)
# ax[1].plot(t/10, n_ind, '|')
# fig.suptitle(title + 'on-off; plot')

# savetitle = title.replace('\n','')
# onofffile = savetitle+'_spon_t'+save_apd+'_%d'%(loop_num)+'.png'
# fig.savefig(onofffile)
# plt.close()

#%%
if get_ani:
    '''spon'''
    #first_stim = 0 #1*n_perStimAmp -1; 
    #last_stim = 0 #1*n_perStimAmp
    start_time = 5000 #data.a1.param.stim1.stim_on[first_stim,0] - 300
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
    
    #stim_on_off = data.a1.param.stim1.stim_on-start_time
    #stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
    
    stim = None #[[[[31.5,31.5]], [stim_on_off], [[6]*stim_on_off.shape[0]]],None]
    #stim = [[[[31.5,31.5],[63.5,-0.5]], [stim_on_off,stim_on_off], [[6]*stim_on_off.shape[0],[6]*stim_on_off.shape[0]]],None]
    
    adpt = None
    #adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    #adpt = [[[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
    ani = fra.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                            frames = frames, start_time = start_time, interval_movie=15, anititle=title,stim=stim, adpt=adpt)
    savetitle = title.replace('\n','')
    
    moviefile = savetitle+'_spon_%d'%loop_num+'.mp4'
    
    # if loop_num%1 == 0:
    ani.save(moviefile)
    
    del ani
    
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
    first_stim = n_StimAmp*n_perStimAmp #1*n_perStimAmp -1 + n_perStimAmp*n_StimAmp; 
    last_stim = n_StimAmp*n_perStimAmp #1*n_perStimAmp + n_perStimAmp*n_StimAmp
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

    adpt = [None, [[[31.5,31.5]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[7]]]]
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








