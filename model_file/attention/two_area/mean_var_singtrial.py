#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:53:50 2021

@author: shni2598
"""

'''
super-poisson of mean-var relation of spike counts
'''
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

data_dir = 'raw_data/'
save_dir = ''#'mean_results/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
analy_type = 'state'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/test_longstim/'+data_dir
sys_argv = int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#loop_num = 0
#%%
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly' #'data_anly' data_anly_temp
save_apd = ''

# onff_method = 'threshold'

# thre_spon = 5#4
# thre_stim = [20] #[12, 15, 30]

# fftplot = 1; getfano = 1
# get_nscorr = 1; get_nscorr_t = 1
# get_TunningCurve = 1; get_HzTemp = 1
# firing_rate_long = 1

# if loop_num%4 == 0: save_img = 1
# else: save_img = 0

# if loop_num%10 ==0: get_ani = 1
# else: get_ani = 0

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']


data = mydata.mydata()
#loop_num = 1
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

if analy_type == 'state': # fbrg: feedback range
    title = '2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                               data.inter.param.peak_p_e2_e1)


#%%
def get_fano_singletrial_multiwin(spk_matrix, analy_dura, win_list, dt=0.1):
    '''
    

    Parameters
    ----------
    spk_matrix : TYPE
        DESCRIPTION.
    analy_dura : TYPE
        DESCRIPTION.
    win_list : TYPE
        DESCRIPTION.
    dt : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    fano_all : TYPE
        DESCRIPTION.

    '''
    fano_all = np.zeros(len(win_list))
    mean_var = []
    
    for win_id in range(len(win_list)):
        print(win_id)
        win = win_list[win_id]
        sample_interval = win
    
        # simu_time_tot = data.param.simutime
        # data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
        
        #analy_dura = np.array([[5000,20000]])
        
        fano_, mean_var_ = fra.get_mean_var_singletrial(spk_matrix, \
                                                               dura=analy_dura, win=win, sample_interval=sample_interval, dt=0.1)
        mean_var.append(mean_var_)
        fano_all[win_id] = np.nanmean(fano_)
    
    return fano_all, mean_var
#%%
# n_StimAmp = 1#data.a1.param.stim1.n_StimAmp
# n_perStimAmp = 20
n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp

stim_amp = [400]

#win_all = np.logspace(0.5,3,20,dtype=int) 
win_all = np.logspace(0.5,4,28,dtype=int)
#win_all = [100]
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)

'''fano-win sens'''
simu_time_tot = data.param.simutime
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

for st in range(len(stim_amp)):

    analy_dura = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
    analy_dura[:,0] += 200

    fano_all_noatt, mean_var_noatt = get_fano_singletrial_multiwin(data.a1.ge.spk_matrix[mua_neuron], \
                                                                 analy_dura, win_list=win_all, dt=0.1)
    
    # data_anly.fano_win = mydata.mydata(); data_anly.fano_win.sens = mydata.mydata()
    # data_anly.fano_win.sens.win_all = win_all
    if st == 0: 
        data_anly.fano_win = mydata.mydata(); data_anly.fano_win.sens = mydata.mydata()
        data_anly.fano_win.sens.win_all = win_all
        data_anly.fano_win.sens.fano_all_noatt = []
        data_anly.fano_win.sens.mean_var_noatt = []
        data_anly.fano_win.sens.fano_all_att = []
        data_anly.fano_win.sens.mean_var_att = []
    data_anly.fano_win.sens.fano_all_noatt.append(fano_all_noatt.copy())
    data_anly.fano_win.sens.mean_var_noatt.append(mean_var_noatt.copy())
    
    
    analy_dura = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
    analy_dura[:,0] += 200

    fano_all_att, mean_var_att = get_fano_singletrial_multiwin(data.a1.ge.spk_matrix[mua_neuron], \
                                                                 analy_dura, win_list=win_all, dt=0.1)

    data_anly.fano_win.sens.fano_all_att.append(fano_all_att.copy())
    data_anly.fano_win.sens.mean_var_att.append(mean_var_att.copy())
    
    '''plot mean-var'''
    plot_mean_var_ind = [4,11,19]
    
    fig, ax = plt.subplots(1,len(plot_mean_var_ind),figsize=[12,8])
    ind_i = 0
    for ind in plot_mean_var_ind:
        
        ax[ind_i].plot(mean_var_noatt[ind][:,0,:].reshape(-1), mean_var_noatt[ind][:,1,:].reshape(-1), '.', c=clr[0], label='win:%d ms; mean:%.2f; noatt'%(win_all[ind],fano_all_noatt[ind]))
        ax[ind_i].plot(mean_var_att[ind][:,0,:].reshape(-1), mean_var_att[ind][:,1,:].reshape(-1), '.', c=clr[1], label='win:%d ms; mean:%.2f; att'%(win_all[ind],fano_all_att[ind]))
        
        ax[ind_i].set_title('win:%.1f'%(win_all[ind]))
        max_lim = max(*ax[ind_i].get_xlim(), *ax[ind_i].get_ylim())
        min_lim = min(*ax[ind_i].get_xlim(), *ax[ind_i].get_ylim())
                        
        ax[ind_i].set_xlim([min_lim,max_lim])
        ax[ind_i].set_ylim([min_lim,max_lim])
        ax[ind_i].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')
        
        ind_i += 1
        
    for axc in ax:
        axc.set_xlabel('mean')
        axc.set_ylabel('var')
        axc.legend()
    
    title_ = title + '\n_st%.1f_sens'%(stim_amp[st])#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    #ax[0].set_title(title_)
    fig.suptitle(title_)
    savetitle = title_.replace('\n','') + '_mv_%d'%(loop_num)+'.png'
    #fanofile = savetitle+'_%d'%(loop_num)+'.png'
    #if save_img: fig.savefig(savetitle)
    fig.savefig(save_dir + savetitle)
    plt.close()

    '''plot fano-win'''
    fig, ax = plt.subplots(1,1,figsize=[8,6])
    ax.plot(win_all, fano_all_noatt, '.', c=clr[0], label='noatt')
    ax.plot(win_all, fano_all_att, '.', c=clr[1], label='att')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('win (ms)')
    ax.set_ylabel('fano')
    ax.set_title('fano-win')
    ax.legend()
    
    title_ = title + '\n_st%.1f_sens'%(stim_amp[st])#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    #ax[0].set_title(title_)
    fig.suptitle(title_)
    savetitle = title_.replace('\n','') + '_fw_%d'%(loop_num)+'.png'
    #fanofile = savetitle+'_%d'%(loop_num)+'.png'
    #if save_img: fig.savefig(savetitle)
    fig.savefig(save_dir + savetitle)
    plt.close()
    
#%%
'''fano-win asso'''

simu_time_tot = data.param.simutime
data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10])


for st in range(len(stim_amp)):
    
    
    analy_dura = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
    analy_dura[:,0] += 200

    fano_all_noatt, mean_var_noatt = get_fano_singletrial_multiwin(data.a2.ge.spk_matrix[mua_neuron], \
                                                                 analy_dura, win_list=win_all, dt=0.1)

    if st == 0: 
        #data_anly.fano_win = mydata.mydata(); 
        data_anly.fano_win.asso = mydata.mydata()
        data_anly.fano_win.asso.win_all = win_all
        data_anly.fano_win.asso.fano_all_noatt = []
        data_anly.fano_win.asso.mean_var_noatt = []
        data_anly.fano_win.asso.fano_all_att = []
        data_anly.fano_win.asso.mean_var_att = []
    data_anly.fano_win.asso.fano_all_noatt.append(fano_all_noatt.copy())
    data_anly.fano_win.asso.mean_var_noatt.append(mean_var_noatt.copy())
      
    # data_anly.fano_win.asso = mydata.mydata()
    # data_anly.fano_win.asso.win_all = win_all
    # data_anly.fano_win.asso.fano_all_noatt = fano_all_noatt
    # data_anly.fano_win.asso.mean_var_noatt = mean_var_noatt
        
    analy_dura = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
    analy_dura[:,0] += 200

    fano_all_att, mean_var_att = get_fano_singletrial_multiwin(data.a2.ge.spk_matrix[mua_neuron], \
                                                                 analy_dura, win_list=win_all, dt=0.1)
    data_anly.fano_win.asso.fano_all_att.append(fano_all_att.copy())
    data_anly.fano_win.asso.mean_var_att.append(mean_var_att.copy())

    # data_anly.fano_win.asso.fano_all_att = fano_all_att
    # data_anly.fano_win.asso.mean_var_att = mean_var_att


    '''plot mean-var'''
    plot_mean_var_ind = [4,11,19]
    
    fig, ax = plt.subplots(2,len(plot_mean_var_ind),figsize=[12,10])
    ind_i = 0
    for ind in plot_mean_var_ind:
        
        ax[0,ind_i].plot(mean_var_noatt[ind][:,0,:].reshape(-1), mean_var_noatt[ind][:,1,:].reshape(-1), '.', c=clr[0], label='win:%d ms; mean:%.2f; noatt'%(win_all[ind],fano_all_noatt[ind]))
        
        ax[0,ind_i].set_title('win:%.1f'%(win_all[ind]))
        max_lim = max(*ax[0,ind_i].get_xlim(), *ax[0,ind_i].get_ylim())
        min_lim = min(*ax[0,ind_i].get_xlim(), *ax[0,ind_i].get_ylim())
                        
        ax[0,ind_i].set_xlim([min_lim,max_lim])
        ax[0,ind_i].set_ylim([min_lim,max_lim])
        ax[0,ind_i].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')
        
        ax[1,ind_i].plot(mean_var_att[ind][:,0,:].reshape(-1), mean_var_att[ind][:,1,:].reshape(-1), '.', c=clr[1], label='win:%d ms; mean:%.2f; att'%(win_all[ind],fano_all_att[ind]))

        ax[1,ind_i].set_title('win:%.1f'%(win_all[ind]))
        max_lim = max(*ax[1,ind_i].get_xlim(), *ax[1,ind_i].get_ylim())
        min_lim = min(*ax[1,ind_i].get_xlim(), *ax[1,ind_i].get_ylim())
                        
        ax[1,ind_i].set_xlim([min_lim,max_lim])
        ax[1,ind_i].set_ylim([min_lim,max_lim])
        ax[1,ind_i].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')
        
        ind_i += 1
    
    for axr in ax:
        for axc in axr:
            axc.set_xlabel('mean')
            axc.set_ylabel('var')
            axc.legend()
    
    title_ = title + '\n_st%.1f_asso'%(stim_amp[st])#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    #ax[0].set_title(title_)
    fig.suptitle(title_)
    savetitle = title_.replace('\n','') + '_mv_%d'%(loop_num)+'.png'
    #fanofile = savetitle+'_%d'%(loop_num)+'.png'
    #if save_img: fig.savefig(savetitle)
    fig.savefig(save_dir + savetitle)
    plt.close()

    '''plot fano-win'''
    fig, ax = plt.subplots(1,1,figsize=[8,6])
    ax.plot(win_all, fano_all_noatt, '.', c=clr[0], label='noatt')
    ax.plot(win_all, fano_all_att, '.', c=clr[1], label='att')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('win (ms)')
    ax.set_ylabel('fano')
    ax.set_title('fano-win')
    ax.legend()
    
    title_ = title + '\n_st%.1f_asso'%(stim_amp[st])#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
    #ax[0].set_title(title_)
    fig.suptitle(title_)
    savetitle = title_.replace('\n','') + '_fw_%d'%(loop_num)+'.png'
    #fanofile = savetitle+'_%d'%(loop_num)+'.png'
    #if save_img: fig.savefig(savetitle)
    fig.savefig(save_dir + savetitle)
    plt.close()
#%%
'''fano-win spon sens'''
analy_dura = np.array([[5000,20000]])

fano_all_spon, mean_var_spon = get_fano_singletrial_multiwin(data.a1.ge.spk_matrix[mua_neuron], \
                                                             analy_dura, win_list=win_all, dt=0.1)


data_anly.fano_win.sens.fano_all_spon = fano_all_spon
data_anly.fano_win.sens.mean_var_spon = mean_var_spon   
 
'''plot mean-var'''
plot_mean_var_ind = [4,11,19]

fig, ax = plt.subplots(1,len(plot_mean_var_ind),figsize=[12,8])
ind_i = 0
for ind in plot_mean_var_ind:
    
    ax[ind_i].plot(mean_var_spon[ind][:,0,:].reshape(-1), mean_var_spon[ind][:,1,:].reshape(-1), '.', c=clr[0], label='win:%d ms; mean:%.2f; spon'%(win_all[ind],fano_all_spon[ind]))
    
    ax[ind_i].set_title('win:%.1f'%(win_all[ind]))
    max_lim = max(*ax[ind_i].get_xlim(), *ax[ind_i].get_ylim())
    min_lim = min(*ax[ind_i].get_xlim(), *ax[ind_i].get_ylim())
                    
    ax[ind_i].set_xlim([min_lim,max_lim])
    ax[ind_i].set_ylim([min_lim,max_lim])
    ax[ind_i].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')
    
    ind_i += 1
    
for axc in ax:
    axc.set_xlabel('mean')
    axc.set_ylabel('var')
    axc.legend()

title_ = title + '\n_spon_sens'#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
#ax[0].set_title(title_)
fig.suptitle(title_)
savetitle = title_.replace('\n','') + '_mv_%d'%(loop_num)+'.png'
#fanofile = savetitle+'_%d'%(loop_num)+'.png'
#if save_img: fig.savefig(savetitle)
fig.savefig(save_dir + savetitle)
plt.close()

'''plot fano-win'''
fig, ax = plt.subplots(1,1,figsize=[8,6])
ax.plot(win_all, fano_all_spon, '.', c=clr[0], label='spon')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('win (ms)')
ax.set_ylabel('fano')
ax.set_title('fano-win')
ax.legend()

title_ = title + '\n_spon_sens'#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
#ax[0].set_title(title_)
fig.suptitle(title_)
savetitle = title_.replace('\n','') + '_fw_%d'%(loop_num)+'.png'
#fanofile = savetitle+'_%d'%(loop_num)+'.png'
#if save_img: fig.savefig(savetitle)
fig.savefig(save_dir + savetitle)
plt.close()
#%%
'''fano-win spon asso'''
analy_dura = np.array([[5000,20000]])

fano_all_spon, mean_var_spon = get_fano_singletrial_multiwin(data.a2.ge.spk_matrix[mua_neuron], \
                                                             analy_dura, win_list=win_all, dt=0.1)
    
data_anly.fano_win.asso.fano_all_spon = fano_all_spon
data_anly.fano_win.asso.mean_var_spon = mean_var_spon   

'''plot mean-var'''
plot_mean_var_ind = [4,11,19]

fig, ax = plt.subplots(1,len(plot_mean_var_ind),figsize=[12,8])
ind_i = 0
for ind in plot_mean_var_ind:
    
    ax[ind_i].plot(mean_var_spon[ind][:,0,:].reshape(-1), mean_var_spon[ind][:,1,:].reshape(-1), '.', c=clr[0], label='win:%d ms; mean:%.2f; spon'%(win_all[ind],fano_all_spon[ind]))
    
    ax[ind_i].set_title('win:%.1f'%(win_all[ind]))
    max_lim = max(*ax[ind_i].get_xlim(), *ax[ind_i].get_ylim())
    min_lim = min(*ax[ind_i].get_xlim(), *ax[ind_i].get_ylim())
                    
    ax[ind_i].set_xlim([min_lim,max_lim])
    ax[ind_i].set_ylim([min_lim,max_lim])
    ax[ind_i].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')
    
    ind_i += 1
    
for axc in ax:
    axc.set_xlabel('mean')
    axc.set_ylabel('var')
    axc.legend()

title_ = title + '\n_spon_asso'#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
#ax[0].set_title(title_)
fig.suptitle(title_)
savetitle = title_.replace('\n','') + '_mv_%d'%(loop_num)+'.png'
#fanofile = savetitle+'_%d'%(loop_num)+'.png'
#if save_img: fig.savefig(savetitle)
fig.savefig(save_dir + savetitle)
plt.close()

'''plot fano-win'''
fig, ax = plt.subplots(1,1,figsize=[8,6])
ax.plot(win_all, fano_all_spon, '.', c=clr[0], label='spon')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('win (ms)')
ax.set_ylabel('fano')
ax.set_title('fano-win')
ax.legend()

title_ = title + '\n_spon_asso'#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
#ax[0].set_title(title_)
fig.suptitle(title_)
savetitle = title_.replace('\n','') + '_fw_%d'%(loop_num)+'.png'
#fanofile = savetitle+'_%d'%(loop_num)+'.png'
#if save_img: fig.savefig(savetitle)
fig.savefig(save_dir + savetitle)
plt.close()
    
#%%
data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)

#%%
# fig, ax = plt.subplots(1,1,figsize=[6,6])
# ax.plot(win_all, fano_all_spon, '.')
# #ax.plot(win_all, fano_all_att, '.')

# ax.set_xscale('log')
# ax.set_yscale('log')
# #%%
# max_lim = max(*ax.get_xlim(), *ax.get_ylim())
# min_lim = min(*ax.get_xlim(), *ax.get_ylim())
                
# ax.set_xlim([min_lim,max_lim])
# ax.set_ylim([min_lim,max_lim])
                
#%%




    #%%

