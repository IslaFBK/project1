#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:04:06 2021

@author: shni2598
"""


import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis
import frequency_analysis as fa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil
#%%
analy_type = 'one_stim'
datapath = ''
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/onearea/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/spon_1area/'
#sys_argv = int(sys.argv[1])
# loop_num = sys_argv #rep_ind*20 + ie_num
# good_dir = 'good/'
# goodsize_dir = 'good_size/'
save_dir = 'results/'

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
#%%
# ie_r_e = np.array([1.6]) #np.arange(1.4,1.61,0.05) #2.76*6.5/5.8*
# ie_r_i = np.array([1.03])#np.arange(0.85,1.14,0.03) #2.450*6.5/5.8*

scale_w_21_e = np.arange(0.8,1.21,0.05)#[1]:#np.arange(0.8,1.21,0.05):
scale_w_21_i = np.arange(0.8,1.21,0.05)
#%%
'''onearea benchmark'''
# databenchmark_path = '/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/onearea/benchmark_hz/'
# data_benmk_one = mydata.mydata()
# data_benmk_one.load(databenchmark_path+'data_benmk%d.file'%1)
# hz_loc_mean_one = data_benmk_one.hz_loc[:,:,0].reshape(5,10,-1).mean(1)
# hz_t_mean_one = data_benmk_one.hz_t
#%%
dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
#fr_bin = np.array([150, 200, 250]) 
fr_bin = np.array([200]) 

N_stim = 60
hz_loc = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
spon_rate1 = 0
hz_t = np.zeros([N_stim, 400])
#trial_per_ie = 10

stim_amp = np.arange(2,5)*200
dist_bin = np.arange(0, 31.5*2**0.5, 2.5)
dist_bin_plot = (dist_bin+(dist_bin[1]-dist_bin[0])*0.5)[:-1]

data_analy = mydata.mydata()
# data_benmk = mydata.mydata()

n_ie = 81#50; 
trial_per_ie = 10
n_amp_stim = 6
n_per_amp = 10
n_amp_stim_att = 3

clr_list = plt.rcParams['axes.prop_cycle'].by_key()['color']


for ie_ind in range(0,n_ie):
    hz_loc = np.zeros([N_stim,len(dist_bin)-1,len(fr_bin)])
    spon_rate1 = 0
    spon_rate2 = 0
    hz_t = np.zeros([N_stim, 400, 2])
    hz_loc_spon = np.zeros([2,len(dist_bin)-1])
    
    for loop_num in range(ie_ind*trial_per_ie, ie_ind*trial_per_ie+trial_per_ie):
        #data_analy = mydata.mydata()
        data_analy.load(datapath+'data_anly%d.file'%loop_num)
        hz_loc += data_analy.hz_loc 
        hz_loc_spon += data_analy.hz_loc_spon
        spon_rate1 += data_analy.spon_rate1
        spon_rate2 += data_analy.spon_rate2
        hz_t += data_analy.hz_t[:,:,:]
        
        
    hz_loc /= trial_per_ie
    spon_rate1 /= trial_per_ie
    spon_rate2 /= trial_per_ie
    hz_t /= trial_per_ie
    hz_loc_spon /= trial_per_ie
    
    for b in range(1):
        hz_loc_mean = hz_loc[:,:,b].reshape(n_amp_stim,n_per_amp,-1).mean(1)
        plt.figure(figsize=[8,6])
        for i in range(n_amp_stim_att):
            
            #plt.plot(dist_bin_plot, hz_loc_mean[i]/(fr_bin[b]/1000), '-o', label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
            plt.plot(dist_bin_plot, hz_loc_mean[i], ls='--', marker='o', c=clr_list[i], label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
        plt.plot(dist_bin_plot, hz_loc_spon[0], ls='--', marker='o', c=clr_list[i+1], label = 'spontaneous')
        
        for i in range(n_amp_stim_att,n_amp_stim):
            
            #plt.plot(dist_bin_plot, hz_loc_mean[i]/(fr_bin[b]/1000), '-o', label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
            plt.plot(dist_bin_plot, hz_loc_mean[i], ls='-', marker='o', c=clr_list[i-n_amp_stim_att], label = 'attention; stim_amp: %.1f Hz'%(stim_amp[i-n_amp_stim_att]))
        plt.plot(dist_bin_plot, hz_loc_spon[1], ls='-', marker='o', c=clr_list[i+1-n_amp_stim_att], label = 'attention; spontaneous')
        
        
        title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
        plt.title('average firing rate versus distance to stimulus centre; 2e1e:%.3f 2e1i%.3f\n Spike-count bin: %.1f ms\n'%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]],fr_bin[b])+title)
        plt.xlim([dist_bin[0],dist_bin[-1]])
        plt.xlabel('distance')
        plt.ylabel('Hz')
        plt.legend()
        #plt.savefig(save_dir+title.replace(':','')+'_countbin%.0f'%fr_bin[b]+'_%d'%ie_ind+'.png')
    
    hz_t_mean = hz_t[:,:, 0].reshape(n_amp_stim,n_per_amp,-1).mean(1)
    plt.figure(figsize=[8,6])
    for i in range(n_amp_stim_att):
        plt.plot(hz_t_mean[i], ls='--', c=clr_list[i], label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
    for i in range(n_amp_stim_att,n_amp_stim):
        plt.plot(hz_t_mean[i], ls='-', c=clr_list[i-n_amp_stim_att], label = 'attention; stim_amp: %.1f Hz'%(stim_amp[i-n_amp_stim_att])) 
    title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
    plt.title(title+'\nsenssory')
    plt.legend()
    plt.savefig(save_dir+title.replace(':','')+'_temp'+'_%d'%ie_ind+'.png')
    
    hz_t_mean = hz_t[:,:, 1].reshape(n_amp_stim,n_per_amp,-1).mean(1)
    plt.figure(figsize=[8,6])
    for i in range(n_amp_stim_att):
        plt.plot(hz_t_mean[i], ls='--', c=clr_list[i], label = 'stim_amp: %.1f Hz'%(stim_amp[i]))
    for i in range(n_amp_stim_att,n_amp_stim):
        plt.plot(hz_t_mean[i], ls='-', c=clr_list[i-n_amp_stim_att], label = 'attention; stim_amp: %.1f Hz'%(stim_amp[i-n_amp_stim_att])) 
    title = "2e1e:%.3f_2e1i:%.3f_1hz:%.2f_2hz:%.2f"%(scale_w_21_e[ie_ind//scale_w_21_i.shape[0]], scale_w_21_i[ie_ind%scale_w_21_i.shape[0]], spon_rate1,spon_rate2)
    plt.title(title+'\nassociation')
    plt.legend()
    plt.savefig(save_dir+title.replace(':','')+'_temp_2'+'_%d'%ie_ind+'.png')
    
    # data_benmk.hz_loc = hz_loc
    # data_benmk.hz_loc_spon = hz_loc_spon
    # data_benmk.hz_t = hz_t_mean
    # data_benmk.spon_rate1 = spon_rate1
    
    #data_benmk.save(data_benmk.class2dict(), datapath+'data_benmk%d.file'%n_ie)

    






