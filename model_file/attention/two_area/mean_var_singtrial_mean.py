#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:31:45 2021

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
save_dir = 'mean_results/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
analy_type = 'state'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/test_longstim/'+data_dir
#sys_argv = int(sys.argv[1])
#loop_num = sys_argv #rep_ind*20 + ie_num
#loop_num = 0
#%%
good_dir = 'good/'
goodsize_dir = 'good_size/'

#savefile_name = 'data_anly' #'data_anly' data_anly_temp
save_apd = ''
data_analy_file = 'data_anly'
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
loop_num = 0
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

# if analy_type == 'state': # fbrg: feedback range
#     title = '2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
#                                                data.inter.param.peak_p_e2_e1)

#%%
# n_StimAmp = 1#data.a1.param.stim1.n_StimAmp
# n_perStimAmp = 20
n_StimAmp = data.a1.param.stim1.n_StimAmp
n_perStimAmp = data.a1.param.stim1.n_perStimAmp

stim_amp = [400]

win_all = np.logspace(0.5,3,20,dtype=int) 
#np.logspace(0.5,3,28,dtype=int)
#win_all = [100]
mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)



n_param = 1
repeat = 20



    
    
    
for param_id in range(n_param):
    
    fano_all_noatt_sens = []
    mean_var_noatt_sens = []
    fano_all_att_sens = []
    mean_var_att_sens = []
    fano_all_spon_sens = []
    mean_var_spon_sens = []
    
    
    fano_all_noatt_asso = []
    mean_var_noatt_asso = []
    fano_all_att_asso = []
    mean_var_att_asso = []
    fano_all_spon_asso = []
    mean_var_spon_asso = []
    
    for loop_num in range(param_id*repeat,(param_id+1)*repeat):
        
        try: 
            data_anly.load(datapath+data_analy_file+'%d.file'%(loop_num))
        except FileNotFoundError:
            print('warning: cannot find %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        except EOFError:
            print('EOFError: file: %s'%(datapath+data_analy_file+'%d.file'%(loop_num)))
            continue
        
        win_all = data_anly.fano_win.sens.win_all #= win_all
        
        fano_all_noatt_sens.append(np.array(data_anly.fano_win.sens.fano_all_noatt))
        mean_var_noatt_sens.append(np.array(data_anly.fano_win.sens.mean_var_noatt))
        fano_all_att_sens.append(np.array(data_anly.fano_win.sens.fano_all_att))
        mean_var_att_sens.append(np.array(data_anly.fano_win.sens.mean_var_att))
        fano_all_spon_sens.append(np.array(data_anly.fano_win.sens.fano_all_spon))
        mean_var_spon_sens.append(np.array(data_anly.fano_win.sens.mean_var_spon))
        
        fano_all_noatt_asso.append(np.array(data_anly.fano_win.asso.fano_all_noatt))
        mean_var_noatt_asso.append(np.array(data_anly.fano_win.asso.mean_var_noatt))
        fano_all_att_asso.append(np.array(data_anly.fano_win.asso.fano_all_att))
        mean_var_att_asso.append(np.array(data_anly.fano_win.asso.mean_var_att))
        fano_all_spon_asso.append(np.array(data_anly.fano_win.asso.fano_all_spon))
        mean_var_spon_asso.append(np.array(data_anly.fano_win.asso.mean_var_spon))        
        
        
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
        title = '2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                                   data.inter.param.peak_p_e2_e1)

    fano_all_noatt_sens_sem = sem(np.array(fano_all_noatt_sens), 0, nan_policy='omit')
    fano_all_noatt_sens_mean = np.nanmean(np.array(fano_all_noatt_sens), 0)#, 0, nan_policy='omit')
    fano_all_att_sens_sem = sem(np.array(fano_all_att_sens), 0, nan_policy='omit')
    fano_all_att_sens_mean = np.nanmean(np.array(fano_all_att_sens), 0)
    fano_all_spon_sens_sem = sem(np.array(fano_all_spon_sens), 0, nan_policy='omit')
    fano_all_spon_sens_mean = np.nanmean(np.array(fano_all_spon_sens), 0)
    
    fano_all_noatt_asso_sem = sem(np.array(fano_all_noatt_asso), 0, nan_policy='omit')
    fano_all_noatt_asso_mean = np.nanmean(np.array(fano_all_noatt_asso), 0)#, 0, nan_policy='omit')
    fano_all_att_asso_sem = sem(np.array(fano_all_att_asso), 0, nan_policy='omit')
    fano_all_att_asso_mean = np.nanmean(np.array(fano_all_att_asso), 0)
    fano_all_spon_asso_sem = sem(np.array(fano_all_spon_asso), 0, nan_policy='omit')
    fano_all_spon_asso_mean = np.nanmean(np.array(fano_all_spon_asso), 0)
    
    mean_var_noatt_sens = np.concatenate(mean_var_noatt_sens, -1)
    mean_var_att_sens = np.concatenate(mean_var_att_sens, -1)
    mean_var_spon_sens = np.concatenate(mean_var_spon_sens, -1)
    
    mean_var_noatt_asso = np.concatenate(mean_var_noatt_asso, -1)
    mean_var_att_asso = np.concatenate(mean_var_att_asso, -1)
    mean_var_spon_asso = np.concatenate(mean_var_spon_asso, -1)
    
    n_data = mean_var_noatt_sens.shape[2]*mean_var_noatt_sens.shape[4]
    choose_neu = np.random.choice(np.arange(n_data), int(n_data/repeat), replace=False)
    
    '''fano-win sens'''
    
    for st in range(len(stim_amp)):
    
        
        '''plot mean-var'''
        plot_mean_var_ind = [4,11,19]
        
        fig, ax = plt.subplots(1,len(plot_mean_var_ind),figsize=[12,6])
        ind_i = 0
        for ind in plot_mean_var_ind:
            
            ax[ind_i].plot(mean_var_noatt_sens[st][ind][:,0,:].reshape(-1)[choose_neu], mean_var_noatt_sens[st][ind][:,1,:].reshape(-1)[choose_neu], \
                           '.', c=clr[0], label='win:%d ms; mean:%.2f; noatt'%(win_all[ind],fano_all_noatt_sens_mean[st][ind]))
            ax[ind_i].plot(mean_var_att_sens[st][ind][:,0,:].reshape(-1)[choose_neu], mean_var_att_sens[st][ind][:,1,:].reshape(-1)[choose_neu], \
                           '.', c=clr[1], label='win:%d ms; mean:%.2f; att'%(win_all[ind],fano_all_att_sens_mean[st][ind]))
            
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
        savetitle = title_.replace('\n','') + '_mv_mean_%d'%(param_id)+'.png'
        #fanofile = savetitle+'_%d'%(loop_num)+'.png'
        #if save_img: fig.savefig(savetitle)
        fig.savefig(save_dir + savetitle)
        plt.close()
    
        '''plot fano-win'''
        fig, ax = plt.subplots(1,1,figsize=[8,6])
        ax.errorbar(win_all, fano_all_noatt_sens_mean[st], fano_all_noatt_sens_sem[st], fmt='-', c=clr[0], marker='o', label='noatt')
        ax.errorbar(win_all, fano_all_att_sens_mean[st], fano_all_att_sens_sem[st], fmt='-', c=clr[1], marker='o', label='att')
        ax.errorbar(win_all, fano_all_spon_sens_mean, fano_all_spon_sens_sem, fmt='-', c=clr[2], marker='o', label='spon')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('win (ms)')
        ax.set_ylabel('fano')
        ax.set_title('fano-win')
        ax.legend()
        
        title_ = title + '\n_st%.1f_sens'%(stim_amp[st])#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
        #ax[0].set_title(title_)
        fig.suptitle(title_)
        savetitle = title_.replace('\n','') + '_fw_mean_%d'%(param_id)+'.png'
        #fanofile = savetitle+'_%d'%(loop_num)+'.png'
        #if save_img: fig.savefig(savetitle)
        fig.savefig(save_dir + savetitle)
        plt.close()
        
    
        '''fano-win asso'''
        
        '''plot mean-var'''
        plot_mean_var_ind = [4,11,19]
        
        fig, ax = plt.subplots(2,len(plot_mean_var_ind),figsize=[12,8])
        ind_i = 0
        for ind in plot_mean_var_ind:
            
            ax[0,ind_i].plot(mean_var_noatt_asso[st][ind][:,0,:].reshape(-1)[choose_neu], mean_var_noatt_asso[st][ind][:,1,:].reshape(-1)[choose_neu], \
                             '.', c=clr[0], label='win:%d ms; mean:%.2f; noatt'%(win_all[ind],fano_all_noatt_asso_mean[st][ind]))
            
            ax[0,ind_i].set_title('win:%.1f'%(win_all[ind]))
            max_lim = max(*ax[0,ind_i].get_xlim(), *ax[0,ind_i].get_ylim())
            min_lim = min(*ax[0,ind_i].get_xlim(), *ax[0,ind_i].get_ylim())
                            
            ax[0,ind_i].set_xlim([min_lim,max_lim])
            ax[0,ind_i].set_ylim([min_lim,max_lim])
            ax[0,ind_i].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')
            
            ax[1,ind_i].plot(mean_var_att_asso[st][ind][:,0,:].reshape(-1)[choose_neu], mean_var_att_asso[st][ind][:,1,:].reshape(-1)[choose_neu], \
                             '.', c=clr[1], label='win:%d ms; mean:%.2f; att'%(win_all[ind],fano_all_att_asso_mean[st][ind]))
    
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
        savetitle = title_.replace('\n','') + '_mv_mean_%d'%(param_id)+'.png'
        #fanofile = savetitle+'_%d'%(loop_num)+'.png'
        #if save_img: fig.savefig(savetitle)
        fig.savefig(save_dir + savetitle)
        plt.close()
    
        '''plot fano-win'''
        fig, ax = plt.subplots(1,1,figsize=[8,6])
        ax.errorbar(win_all, fano_all_noatt_asso_mean[st], fano_all_noatt_asso_sem[st], fmt='-', c=clr[0], marker='o', label='noatt')
        ax.errorbar(win_all, fano_all_att_asso_mean[st], fano_all_att_asso_sem[st], fmt='-', c=clr[1], marker='o', label='att')
        ax.errorbar(win_all, fano_all_spon_asso_mean, fano_all_spon_asso_sem, fmt='-', c=clr[2], marker='o', label='spon')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('win (ms)')
        ax.set_ylabel('fano')
        ax.set_title('fano-win')
        ax.legend()
        
        title_ = title + '\n_st%.1f_asso'%(stim_amp[st])#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
        #ax[0].set_title(title_)
        fig.suptitle(title_)
        savetitle = title_.replace('\n','') + '_fw_mean_%d'%(param_id)+'.png'
        #fanofile = savetitle+'_%d'%(loop_num)+'.png'
        #if save_img: fig.savefig(savetitle)
        fig.savefig(save_dir + savetitle)
        plt.close()
        #%%
        '''fano-win spon sens''' 
         
        '''plot mean-var'''
        plot_mean_var_ind = [4,11,19]
        
        fig, ax = plt.subplots(1,len(plot_mean_var_ind),figsize=[12,6])
        ind_i = 0
        for ind in plot_mean_var_ind:
            
            ax[ind_i].plot(mean_var_spon_sens[ind][:,0,:].reshape(-1), mean_var_spon_sens[ind][:,1,:].reshape(-1), \
                           '.', c=clr[0], label='win:%d ms; mean:%.2f; spon'%(win_all[ind],fano_all_spon_sens_mean[ind]))
            
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
        savetitle = title_.replace('\n','') + '_mv_mean_%d'%(param_id)+'.png'
        #fanofile = savetitle+'_%d'%(loop_num)+'.png'
        #if save_img: fig.savefig(savetitle)
        fig.savefig(save_dir + savetitle)
        plt.close()

        #%%
        '''fano-win spon asso'''

        '''plot mean-var'''
        plot_mean_var_ind = [4,11,19]
        
        fig, ax = plt.subplots(1,len(plot_mean_var_ind),figsize=[12,6])
        ind_i = 0
        for ind in plot_mean_var_ind:
            
            #ax[ind_i].plot(mean_var_spon[ind][:,0,:].reshape(-1), mean_var_spon[ind][:,1,:].reshape(-1), '.', c=clr[0], label='win:%d ms; mean:%.2f; spon'%(win_all[ind],fano_all_spon[ind]))
            ax[ind_i].plot(mean_var_spon_asso[ind][:,0,:].reshape(-1), mean_var_spon_asso[ind][:,1,:].reshape(-1), \
                           '.', c=clr[0], label='win:%d ms; mean:%.2f; spon'%(win_all[ind],fano_all_spon_asso_mean[ind]))
            
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
        savetitle = title_.replace('\n','') + '_mv_mean_%d'%(param_id)+'.png'
        #fanofile = savetitle+'_%d'%(loop_num)+'.png'
        #if save_img: fig.savefig(savetitle)
        fig.savefig(save_dir + savetitle)
        plt.close()
        
        
        
    #%%
    #data_anly.save(data_anly.class2dict(), datapath+savefile_name+'%d.file'%loop_num)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    
    
    
        #%