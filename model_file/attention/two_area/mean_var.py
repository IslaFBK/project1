#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:20:20 2021

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



data_dir = 'raw_data/'
save_dir = 'mean_results/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
analy_type = 'state'
datapath = data_dir
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/on_off/test2/'+data_dir
#sys_argv = int(sys.argv[1])
#loop_num = sys_argv #rep_ind*20 + ie_num
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
loop_num = 0
data.load(datapath+'data%d.file'%loop_num)

mua_loca = [0, 0]
mua_range = 5 
mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)

start_time_ls = [0, 200, 500] # time point begin to count spike
anly_dura_ls = [200, 500, 1000] # length of duration to count spike

n_StimAmp = 1#data.a1.param.stim1.n_StimAmp
n_perStimAmp = 50
stim_amp = [400]

m_v_1 = np.zeros([n_StimAmp, len(mua_neuron),2,20])
m_v_att_1 = np.zeros([n_StimAmp, len(mua_neuron),2,20])

m_v_2 = np.zeros([n_StimAmp, len(mua_neuron),2,20])
m_v_att_2 = np.zeros([n_StimAmp, len(mua_neuron),2,20])
    

n_param = 1
repeat = 20

for param_id in range(n_param):
    for start_time in start_time_ls:
        for anly_dura  in anly_dura_ls:
            print('param id: %d, start_time: %.1f, anly_dura: %.1f' %(param_id,start_time,anly_dura))
            m_v_1[:] = np.nan
            m_v_att_1[:] = np.nan 
            m_v_2[:] = np.nan
            m_v_att_2[:] = np.nan
            
            for loop_num in range(param_id*repeat,(param_id+1)*repeat):
                
                data.load(datapath+'data%d.file'%loop_num)
                
                if analy_type == 'state': # fbrg: feedback range
                    title = '2irie%.2f_2ndgk%.1f_pk2e1e%.2f'%(data.param.ie_r_i2, data.param.new_delta_gk_2, \
                                                               data.inter.param.peak_p_e2_e1)
                #data_anly = mydata.mydata()
               
                # start = 5000; end = 20000
                # analy_dura = np.array([[start,end]])
                
                simu_time_tot = data.param.simutime
                
                
                
                #n = 0; 
                
                
                for st in range(n_StimAmp):
                    '''sens no att'''
                    data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
                    analy_dura = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
                    analy_dura[:,0] += start_time
                    analy_dura[:,1] = analy_dura[:,0] + anly_dura
                        
                    m_v_1[st, :,:,loop_num] = fra.get_mean_var(data.a1.ge.spk_matrix[mua_neuron], dura=analy_dura, dt=0.1)
                    '''sens att'''
                    analy_dura = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
                    analy_dura[:,0] += start_time
                    analy_dura[:,1] = analy_dura[:,0] + anly_dura
                        
                    m_v_att_1[st, :,:,loop_num] = fra.get_mean_var(data.a1.ge.spk_matrix[mua_neuron], dura=analy_dura, dt=0.1)    
               
                    '''asso no att'''
                    data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10])
                    
                    #st = 0;
                    analy_dura = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp].copy()
                    analy_dura[:,0] += start_time
                    analy_dura[:,1] = analy_dura[:,0] + anly_dura
                        
                    m_v_2[st, :,:,loop_num] = fra.get_mean_var(data.a2.ge.spk_matrix[mua_neuron], dura=analy_dura, dt=0.1)
                    '''asso att'''
                    analy_dura = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp].copy()
                    analy_dura[:,0] += start_time
                    analy_dura[:,1] = analy_dura[:,0] + anly_dura
                        
                    m_v_att_2[st, :,:,loop_num] = fra.get_mean_var(data.a2.ge.spk_matrix[mua_neuron], dura=analy_dura, dt=0.1) 
            print('plot results')
            for st in range(n_StimAmp):    
                fig, ax = plt.subplots(2,2,figsize=[12,12])
                '''sens'''
                ax[0,0].plot(m_v_1[st,:,0,:].reshape(-1), m_v_1[st,:,1,:].reshape(-1), '.', c=clr[0], label='sens; no-att')
                ax[0,0].plot(m_v_att_1[st,:,0,:].reshape(-1), m_v_att_1[st,:,1,:].reshape(-1), '.', c=clr[1], label='sens; att')
                               
                max_lim = max(*ax[0,0].get_xlim(), *ax[0,0].get_ylim())
                min_lim = min(*ax[0,0].get_xlim(), *ax[0,0].get_ylim())
                
                ax[0,0].set_xlim([min_lim,max_lim])
                ax[0,0].set_ylim([min_lim,max_lim])
                
                ax[0,0].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')
                '''asso'''
                ax[0,1].plot(m_v_2[st,:,0,:].reshape(-1), m_v_2[st,:,1,:].reshape(-1), '.', c=clr[0], label='asso; no-att')
                ax[0,1].plot(m_v_att_2[st,:,0,:].reshape(-1), m_v_att_2[st,:,1,:].reshape(-1), '.', c=clr[1], label='asso; att')
                               
                max_lim = max(*ax[0,1].get_xlim(), *ax[0,1].get_ylim())
                min_lim = min(*ax[0,1].get_xlim(), *ax[0,1].get_ylim())
                
                ax[0,1].set_xlim([min_lim,max_lim])
                ax[0,1].set_ylim([min_lim,max_lim])
                
                ax[0,1].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')                
                '''asso no att'''
                ax[1,0].plot(m_v_2[st,:,0,:].reshape(-1), m_v_2[st,:,1,:].reshape(-1), '.', c=clr[0], label='asso; no-att')
                #ax[0,1].plot(m_v_att_2[st,:,0,:].reshape(-1), m_v_att_2[st,:,1,:].reshape(-1), '.', c=clr[1], label='asso; att')
                               
                max_lim = max(*ax[1,0].get_xlim(), *ax[1,0].get_ylim())
                min_lim = min(*ax[1,0].get_xlim(), *ax[1,0].get_ylim())
                
                ax[1,0].set_xlim([min_lim,max_lim])
                ax[1,0].set_ylim([min_lim,max_lim])
                
                ax[1,0].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')   
                '''asso att'''
                #ax[0,1].plot(m_v_2[st,:,0,:].reshape(-1), m_v_2[st,:,1,:].reshape(-1), '.', c=clr[0], label='asso; no-att')
                ax[1,1].plot(m_v_att_2[st,:,0,:].reshape(-1), m_v_att_2[st,:,1,:].reshape(-1), '.', c=clr[1], label='asso; att')
                               
                max_lim = max(*ax[1,1].get_xlim(), *ax[1,1].get_ylim())
                min_lim = min(*ax[1,1].get_xlim(), *ax[1,1].get_ylim())
                
                ax[1,1].set_xlim([min_lim,max_lim])
                ax[1,1].set_ylim([min_lim,max_lim])
                
                ax[1,1].plot([min_lim, max_lim],[min_lim, max_lim], ls='--')   
                
                for axr in ax: 
                    for axc in axr:
                        axc.set_xlabel('mean')
                        axc.set_ylabel('var')
                        axc.legend()
                
                title_ = title + '\n_st%.1f_start%.1f_dura%.1f'%(stim_amp[st], start_time, anly_dura)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
                #ax[0].set_title(title_)
                fig.suptitle(title_)
                savetitle = title_.replace('\n','') + '_%d'%(param_id)+'.png'
                #fanofile = savetitle+'_%d'%(loop_num)+'.png'
                #if save_img: fig.savefig(savetitle)
                fig.savefig(save_dir + savetitle)
                plt.close()



#%%        



    
# #%%
# m_v_2 = np.zeros([len(mua_neuron),2,20])
# m_v_att_2 = np.zeros([len(mua_neuron),2,20])
    
# n_StimAmp = 1#data.a1.param.stim1.n_StimAmp


# for loop_num in range(20):
    
#     data.load(datapath+'data%d.file'%loop_num)
#     print('%s'%loop_num)
#     #data_anly = mydata.mydata()
   
#     # start = 5000; end = 20000
#     # analy_dura = np.array([[start,end]])
    
#     simu_time_tot = data.param.simutime
#     data.a2.ge.get_sparse_spk_matrix([data.a2.param.Ne, simu_time_tot*10])
    
#     n = 0; n_perStimAmp = 50
#     analy_dura = data.a1.param.stim1.stim_on[n*n_perStimAmp:(n+1)*n_perStimAmp].copy()
#     analy_dura[:,0] += 500
#     analy_dura[:,1] = analy_dura[:,0] + 200
        
#     m_v_2[:,:,loop_num] = get_mean_var(data.a2.ge.spk_matrix[mua_neuron], dura=analy_dura, dt=0.1)
    
#     analy_dura = data.a1.param.stim1.stim_on[(n+n_StimAmp)*n_perStimAmp:(n+n_StimAmp+1)*n_perStimAmp].copy()
#     analy_dura[:,0] += 500
#     analy_dura[:,1] = analy_dura[:,0] + 200
        
#     m_v_att_2[:,:,loop_num] = get_mean_var(data.a2.ge.spk_matrix[mua_neuron], dura=analy_dura, dt=0.1)    
     
    
# #%%

# plt.figure()
# plt.plot(m_v[:,0,:].reshape(-1), m_v[:,1,:].reshape(-1), '.')
# plt.plot(m_v_att[:,0,:].reshape(-1), m_v_att[:,1,:].reshape(-1), '.')

# plt.plot(np.arange(10),np.arange(10), ls='--')
# plt.xlim([0,12])
# plt.ylim([0,12])
# #%%
# plt.figure()
# #plt.plot(m_v_2[:,0,:].reshape(-1), m_v_2[:,1,:].reshape(-1), '.')
# plt.plot(m_v_att[:,0,:].reshape(-1), m_v_att[:,1,:].reshape(-1), '.')
# plt.plot(np.arange(10),np.arange(10))
# plt.xlim([0,12])
# plt.ylim([0,12])

# plt.plot(m_v_att_2[:,0,:].reshape(-1), m_v_att_2[:,1,:].reshape(-1), '.')



# #%%
# def get_mean_var(spk_mat, dura, dt=0.1):
#     '''
    

#     Parameters
#     ----------
#     spk_mat : sparse matrix
#         spike time and neuron index.
#     dura : 2-D array
#         start and end time of analysis period of each response.
#     dt : scalar (ms)
#         simulation time step. The default is 0.1 ms.

#     Returns
#     -------
#     mean_var : 2-D array
#         first colum: mean; second colum: variance.

#     '''
#     spk_mat = spk_mat.tocsc()
    
#     mean_var = np.zeros([spk_mat.shape[0], 2])
#     resp = np.zeros([spk_mat.shape[0] , dura.shape[0]])
#     dt_ = int(round(1/dt))
    
#     for i in range(dura.shape[0]):
        
#         resp[:,i] = spk_mat[:, dura[i,0]*dt_:dura[i,1]*dt_].sum(1).A.reshape(-1)
    
#     mean_var[:,0] = resp.mean(1)
#     mean_var[:,1] = resp.var(1)
    
#     return mean_var
    
    


    #%%

