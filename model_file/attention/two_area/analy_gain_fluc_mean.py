#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:54:11 2021

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
#import firing_rate_analysis as fra
#import frequency_analysis as fqa
#import fano_mean_match
#import find_change_pts
import connection as cn
#import pickle
import sys
import os
import matplotlib.pyplot as plt
#import shutil

#from scipy import optimize

import gain_fluctuation as gfluc
#%%
analy_type = 'fbrg'

save_dir = 'mean_results/'
data_analy_file = 'data_anly_gain' #data_anly'#data_anly' #data_anly_fano data_anly_fano10hz data_anly_nc data_anly_fano10hz data_anly_fanorange
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
# n_in_bin = [None]*(dist_bin.shape[0]-1)
# neuron = np.arange(data.a1.param.Ne)

# for i in range(len(dist_bin)-1):
#     n_in_bin[i] = neuron[(dist >= dist_bin[i]) & (dist < dist_bin[i+1])]

n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp
stim_amp = 200*2**np.arange(n_StimAmp)

repeat = 20

#%%

# addi_q_index_noatt = [] data_anly.addi.q_index_noatt #= q_addi_noatt
# multip_q_index_noatt = [] data_anly.multip.q_index_noatt #= q_mutp_noatt
# affine_q_index_noatt = [] data_anly.affine.q_index_noatt# = q_affi_noatt

# addi_q_index_att = [] data_anly.addi.q_index_att# = q_addi_att
# multip_q_index_att = [] data_anly.multip.q_index_att# = q_mutp_att
# affine_q_index_att = [] data_anly.affine.q_index_att# = q_affi_att

# addifluc_std = [] data_anly.addi.fluc_std #= np.zeros([n_StimAmp, 2])
# multip_fluc_std = [] data_anly.multip.fluc_std #= np.zeros([n_StimAmp, 2])
# affine_fluc_multip_std = [] data_anly.affine.fluc_multip_std #= np.zeros([n_StimAmp, 2])
# affine_fluc_addi_std = [] data_anly.affine.fluc_addi_std #= np.zeros([n_StimAmp, 2])


#%%
for param_i in range(0, 1):
    addi_q_index_noatt = [] #data_anly.addi.q_index_noatt #= q_addi_noatt
    multip_q_index_noatt = [] #data_anly.multip.q_index_noatt #= q_mutp_noatt
    affine_q_index_noatt = [] #data_anly.affine.q_index_noatt# = q_affi_noatt
    
    addi_q_index_att = [] #data_anly.addi.q_index_att# = q_addi_att
    multip_q_index_att = [] #data_anly.multip.q_index_att# = q_mutp_att
    affine_q_index_att = [] #data_anly.affine.q_index_att# = q_affi_att
    
    addi_fluc_std = [] #data_anly.addi.fluc_std #= np.zeros([n_StimAmp, 2])
    multip_fluc_std = [] #data_anly.multip.fluc_std #= np.zeros([n_StimAmp, 2])
    affine_fluc_multip_std = [] #data_anly.affine.fluc_multip_std #= np.zeros([n_StimAmp, 2])
    affine_fluc_addi_std = [] #data_anly.affine.fluc_addi_std #= np.zeros([n_StimAmp, 2])
            
        # nc_t[:] = np.nan

    for loop_num in range(param_i*repeat, (param_i+1)*repeat):
        try: 
            data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        except EOFError:
            print('EOFError: file: %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        addi_q_index_noatt.append(data_anly.addi.q_index_noatt) #= q_addi_noatt
        multip_q_index_noatt.append(data_anly.multip.q_index_noatt) #= q_mutp_noatt
        affine_q_index_noatt.append(data_anly.affine.q_index_noatt)# = q_affi_noatt
        
        addi_q_index_att.append(data_anly.addi.q_index_att)# = q_addi_att
        multip_q_index_att.append(data_anly.multip.q_index_att)# = q_mutp_att
        affine_q_index_att.append(data_anly.affine.q_index_att)# = q_affi_att
        
        addi_fluc_std.append(data_anly.addi.fluc_std) #= np.zeros([n_StimAmp, 2])
        multip_fluc_std.append(data_anly.multip.fluc_std) #= np.zeros([n_StimAmp, 2])
        affine_fluc_multip_std.append(data_anly.affine.fluc_multip_std) #= np.zeros([n_StimAmp, 2])
        affine_fluc_addi_std.append(data_anly.affine.fluc_addi_std) #= np.zeros([n_StimAmp, 2])
    
    addi_q_index_noatt = np.array(addi_q_index_noatt)
    multip_q_index_noatt = np.array(multip_q_index_noatt)#.append(data_anly.multip.q_index_noatt) #= q_mutp_noatt
    affine_q_index_noatt = np.array(affine_q_index_noatt) #.append(data_anly.affine.q_index_noatt)# = q_affi_noatt
    
    addi_q_index_att = np.array(addi_q_index_att)#.append(data_anly.addi.q_index_att)# = q_addi_att
    multip_q_index_att = np.array(multip_q_index_att)#.append(data_anly.multip.q_index_att)# = q_mutp_att
    affine_q_index_att = np.array(affine_q_index_att)#.append(data_anly.affine.q_index_att)# = q_affi_att
    
    addi_fluc_std = np.array(addi_fluc_std)#.append(data_anly.addi.fluc_std) #= np.zeros([n_StimAmp, 2])
    multip_fluc_std = np.array(multip_fluc_std)#.append(data_anly.multip.fluc_std) #= np.zeros([n_StimAmp, 2])
    affine_fluc_multip_std = np.array(affine_fluc_multip_std)#.append(data_anly.affine.fluc_multip_std) #= np.zeros([n_StimAmp, 2])
    affine_fluc_addi_std = np.array(affine_fluc_addi_std)#.append(data_anly.affine.fluc_addi_std) #= np.zeros([n_StimAmp, 2])    
    
    found = False
    cannotfind = False
    file_id = loop_num
    while not found:
        if file_id < param_i*repeat:
            print('cannot find any data file for param: %d'%param_i)
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
    
    #%%
    if analy_type == 'fbrg': # fbrg: feedback range
        title = 'hz_2irie%.2f_2ndgk%.1f_2e1ir%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                                   data.inter.param.tau_p_d_e2_i1)
    
    '''q index'''
    fig, ax = plt.subplots(2,3, figsize=[10,8])
    
    for i in range(repeat):
        ax[0,0].scatter(multip_q_index_noatt[i], addi_q_index_noatt[i])
        _, p_ = scipy.stats.wilcoxon(multip_q_index_noatt.reshape(-1) - addi_q_index_noatt.reshape(-1),alternative='greater')
        ax[0,0].text(0.5,0.9,'%.3f'%p_)
        ax[0,1].scatter(affine_q_index_noatt[i], addi_q_index_noatt[i])
        _, p_ = scipy.stats.wilcoxon(affine_q_index_noatt.reshape(-1) - addi_q_index_noatt.reshape(-1),alternative='greater')
        ax[0,1].text(0.5,0.9,'%.3f'%p_)
        ax[0,2].scatter(affine_q_index_noatt[i], multip_q_index_noatt[i])
        _, p_ = scipy.stats.wilcoxon(affine_q_index_noatt.reshape(-1) - multip_q_index_noatt.reshape(-1),alternative='greater')
        ax[0,2].text(0.5,0.9,'%.3f'%p_)
        ax[1,0].scatter(multip_q_index_att[i], addi_q_index_att[i])
        _, p_ = scipy.stats.wilcoxon(multip_q_index_att.reshape(-1) - addi_q_index_att.reshape(-1),alternative='greater')
        ax[1,0].text(0.5,0.9,'%.3f'%p_)
        ax[1,1].scatter(affine_q_index_att[i], addi_q_index_att[i])
        _, p_ = scipy.stats.wilcoxon(affine_q_index_att.reshape(-1) - addi_q_index_att.reshape(-1),alternative='greater')
        ax[1,1].text(0.5,0.9,'%.3f'%p_)
        ax[1,2].scatter(affine_q_index_att[i], multip_q_index_att[i])
        _, p_ = scipy.stats.wilcoxon(affine_q_index_att.reshape(-1) - multip_q_index_att.reshape(-1),alternative='greater')
        ax[1,2].text(0.5,0.9,'%.3f'%p_)
    
    ax[0,1].set_title('p value indicates if model in x axis better than that in y axis')
    
    for axy in ax:
        for axxi, axx in enumerate(axy):
            if axxi == 0: axx.set_xlabel('multip'); axx.set_ylabel('addi'); 
            elif axxi == 1: axx.set_xlabel('affine'); axx.set_ylabel('addi'); 
            elif axxi == 2: axx.set_xlabel('affine'); axx.set_ylabel('multip'); 
                
            axx.plot([-0.5,1.1],[-0.5,1.1],'r--')
            axx.set_xlim([-0.5,1.1])
            axx.set_ylim([-0.5,1.1])
    
    title_ = title + '\n_quality_index'#%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    fig.suptitle(title_)
    savetitle = title_.replace('\n','')
    savetitle = savetitle+'_%d'%(param_i)+'.png'
    #if save_img: 
    fig.savefig(save_dir + savetitle)
    plt.close()
    #%%
    '''fluc modulation'''
    p_fluc = np.zeros([4, len(stim_amp)])
    
    for st in range(len(data_anly.addi.fluc_std)):
        _, p_fluc[0,st] = scipy.stats.wilcoxon(addi_fluc_std[:, st, 1] - addi_fluc_std[:, st, 0])
        _, p_fluc[1,st] = scipy.stats.wilcoxon(multip_fluc_std[:, st, 1] - multip_fluc_std[:, st, 0])
        _, p_fluc[2,st] = scipy.stats.wilcoxon(affine_fluc_multip_std[:, st, 1] - affine_fluc_multip_std[:, st, 0])
        _, p_fluc[3,st] = scipy.stats.wilcoxon(affine_fluc_addi_std[:, st, 1] - affine_fluc_addi_std[:, st, 0])
        
    fig, ax = plt.subplots(4,1, figsize=[6,8])
    for ii in range(repeat):
        for st in range(len(data_anly.addi.fluc_std)):
        #for ii, std in enumerate(data_anly.addi.fluc_std):
            if ii == 0: ax[0].plot(np.arange(2)+st*2, addi_fluc_std[ii][st], c=clr[st], label='st:%d'%stim_amp[st])
            else: ax[0].plot(np.arange(2)+st*2, addi_fluc_std[ii][st], c=clr[st])
            ax[1].plot(np.arange(2)+st*2, multip_fluc_std[ii][st], c=clr[st])
            ax[2].plot(np.arange(2)+st*2, affine_fluc_multip_std[ii][st], c=clr[st])
            ax[3].plot(np.arange(2)+st*2, affine_fluc_addi_std[ii][st], c=clr[st])
        
    ax[0].set_title('addi'); ax[0].xaxis.set_visible(False)
    ax[1].set_title('multip'); ax[1].xaxis.set_visible(False)
    ax[2].set_title('affine multip'); ax[2].xaxis.set_visible(False)
    ax[3].set_title('affine addi'); ax[3].xaxis.set_visible(False)
    
    for axi in range(len(ax)):
        for st in range(len(data_anly.addi.fluc_std)):
            ax[axi].text(0.9+st*2, 0.9*ax[axi].get_ylim()[1], '%.3f'%p_fluc[axi,st])
        
    ax[0].legend()
    
    title_ = title + '\n_fluc_modu'#%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    fig.suptitle(title_)
    savetitle = title_.replace('\n','')
    savetitle = savetitle+'_%d'%(param_i)+'.png'
    #if save_img: 
    fig.savefig(save_dir + savetitle)
    plt.close()








































