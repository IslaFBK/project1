#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:08:28 2021

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
import find_change_pts
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import shutil

from scipy import optimize

import gain_fluctuation
#%%
data_dir = 'raw_data/'
analy_type = 'state'
datapath = data_dir
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/state/param1/'+data_dir

sys_argv = 0 #int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
#loop_num = 0

good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly_onoff' #'data_anly' data_anly_temp
save_apd = ''

onff_method = 'threshold'

thre_spon = 5#4
thre_stim = [20] #[12, 15, 30]

fftplot = 1; getfano = 1
get_nscorr = 1; get_nscorr_t = 1
get_TunningCurve = 1; get_HzTemp = 1
firing_rate_long = 1

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


#%%
#mua_loca = [0, 0]
mua_range = 2 
#mua_neuron = cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, mua_loca, mua_range, data.a1.param.width)
#%%
record_neugroup = []
elec_posi_x = np.arange(-12, 12.1, 4)
elec_posi = np.zeros([elec_posi_x.shape[0], 2])
elec_posi[:,0] = elec_posi_x
for elec in elec_posi:
    record_neugroup.append(cn.findnearbyneuron.findnearbyneuron(data.a1.param.e_lattice, elec, mua_range, data.a1.param.width))
#%%

plt.figure()
plt.scatter(data.a1.param.e_lattice[:,0], data.a1.param.e_lattice[:,1],s=1.5)
for elec_n in range(len(record_neugroup)):
    #
    plt.scatter(data.a1.param.e_lattice[record_neugroup[elec_n]][:,0], data.a1.param.e_lattice[record_neugroup[elec_n]][:,1],s=1.5)
#%%

simu_time_tot = data.param.simutime
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

#%%
st = 1

MUA_muti = multiMUA_crosstrial(data.a1.ge.spk_matrix, data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp], record_neugroup)



u, s, vh = np.linalg.svd(MUA_muti, full_matrices=False)


approx_MUA = s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))

fluc = approx_MUA[0,:]/approx_MUA[0,:].mean()


#st = 0
MUA_muti_att = multiMUA_crosstrial(data.a1.ge.spk_matrix, data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp], record_neugroup)



u_att, s_att, vh_att = np.linalg.svd(MUA_muti_att, full_matrices=False)


approx_MUA_att = s_att[0]*np.dot(u_att[:,0].reshape(-1,1), vh_att[0,:].reshape(1,-1))

fluc_att = approx_MUA_att[0,:]/approx_MUA_att[0,:].mean()

#%%
gain_m = gain_fluctuation.get_multiplicative_fluc(data.a1.ge.spk_matrix, data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(),\
                                         record_neugroup, stim_indptr)
#%%
gain_m.fluc_multip[50:100]


#%%
spon_dura_tmp = np.arange(5000,20001-250,250, dtype=int)
spon_dura = np.zeros([len(spon_dura_tmp),2], int)
spon_dura[:,0] = spon_dura_tmp
spon_dura[:,1] = spon_dura_tmp + 250
del spon_dura_tmp

MUA_muti_spon = multiMUA_crosstrial(data.a1.ge.spk_matrix, spon_dura, record_neugroup)



u_spon, s_spon, vh_spon = np.linalg.svd(MUA_muti_spon, full_matrices=False)


approx_MUA_spon = s_spon[0]*np.dot(u_spon[:,0].reshape(-1,1), vh_spon[0,:].reshape(1,-1))

fluc_spon = approx_MUA_spon[0,:]/approx_MUA_spon[0,:].mean()



#%%
plt.figure()
plt.plot(fluc)
plt.plot(fluc_att)



#%%
plt.figure()
plt.plot(approx_MUA.mean(1))


#%%

plt.figure()

for trial in range(20):
    plt.plot(MUA_muti_spon[:,trial])
    
#%%
plt.figure()
plt.plot(MUA_muti_spon[:,:].mean(1))
    

#%%

def multiMUA_crosstrial(spk_mat, dura, record_neugroup, dt=0.1):
    
    dt_ = int(round(1/dt))
    spk_mat = spk_mat.tocsr()
    
    MUA = np.zeros([len(record_neugroup), len(dura)])
    for neu_i, neu in enumerate(record_neugroup):
        for stim_i, stim_dura in enumerate(dura):
            MUA[neu_i, stim_i] = spk_mat[neu, stim_dura[0]*dt_:stim_dura[1]*dt_].sum()
            
    return MUA
    
#%%
dura_spon_st_noatt = np.vstack((spon_dura, data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp]))
dura_spon_st_att = np.vstack((spon_dura, data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:]))


#%%
stim_indptr = np.arange(0, n_StimAmp*n_perStimAmp+1, n_perStimAmp)
stim_indptr = np.hstack(([0], spon_dura.shape[0]+stim_indptr))
#%%

def get_addititive_fluc(spk_mat, dura, record_neugroup, stim_indptr):
    
    MUA_multi = multiMUA_crosstrial(spk_mat, dura, record_neugroup)
    mean_MUA = np.zeros([MUA_multi.shape[0], stim_indptr.shape[0]-1])
    for st in range(len(stim_indptr)-1):
        mean_MUA[:,st] = MUA_multi[:, stim_indptr[st]:stim_indptr[st+1]].mean(1)
        
    mean_MUA_mat = np.zeros(MUA_multi.shape)
    for st in range(len(stim_indptr)-1):
        mean_MUA_mat[:, stim_indptr[st]:stim_indptr[st+1]] = np.tile(mean_MUA[:,st].reshape(-1,1), stim_indptr[st+1]-stim_indptr[st])
    
    u, s, vh = np.linalg.svd((MUA_multi - mean_MUA_mat), full_matrices=False)
    approx = s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))

    #fluc_addi = approx[0,:]/approx[0,:].mean()
    fluc_addi = approx[0,:]/approx[0,:].std()
    #coup_addi = approx.mean(1)
    coup_addi = approx.std(1)
    
    R = mydata.mydata()
    R.fluc_addi = fluc_addi 
    R.coup_addi = coup_addi
    R.mean_MUA = mean_MUA
    R.mean_MUA_mat =  mean_MUA_mat
    R.MUA_multi = MUA_multi
    R.u = u
    R.s = s
    R.vh = vh
    
    return R
    #return fluc_addi, coup_addi, mean_MUA, mean_MUA_mat, MUA_multi, u, s, vh

#%%
addi_spon_st = \
    get_addititive_fluc(data.a1.ge.spk_matrix, dura_spon_st_noatt, record_neugroup, stim_indptr)   
#%%
stim_indptr_st_noatt = np.arange(0, n_StimAmp*n_perStimAmp+1, n_perStimAmp)

addi_st_noatt = \
    get_addititive_fluc(data.a1.ge.spk_matrix, \
                        data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp], record_neugroup, stim_indptr_st_noatt)   


#%%
resi_mat = MUA_multi - mean_MUA_mat


#%%
MUA_muti_spon = multiMUA_crosstrial(data.a1.ge.spk_matrix, spon_dura, record_neugroup)


#%%
x = np.array([1,2,3,4,3])

xProx = np.array([0,1,3,4,2])

find_fluc = lambda k, x, xProx: np.sum((x - k*xProx)**2)
#%%

#%%
minimum = optimize.fmin(find_fluc, 1, (x, xProx),disp=True)

#%%

def xvalid_addititive_fluc(spk_mat, dura, record_neugroup, stim_indptr):
    
    MUA_multi = multiMUA_crosstrial(spk_mat, dura, record_neugroup)
    
    #select_trial = np.ones(MUA_multi.shape[1],dtype=bool)
    
    MUA_multi_train = MUA_multi[:,1:].copy()
    
    MUA_multi_predict = np.zeros(MUA_multi.shape)
    
    tst_t_sum = 0
    for tst_t in range(MUA_multi.shape[1]):
        #print('test_trial: ',tst_t)
        if tst_t > 0:
            MUA_multi_train[:,tst_t-1] = MUA_multi[:,tst_t-1]
            #if tst_t < 10:
                #print(MUA_multi_train[:,:10])
        # if tst_t == 0:
        #     select_trial[tst_t] = False
        # else:
        #     select_trial[tst_t] = False
        #     select_trial[tst_t-1] = True
        
        for ii in range(1,len(stim_indptr)):
            if tst_t < stim_indptr[ii]:
                stim_indptr_train = stim_indptr.copy()
                stim_indptr_train[ii:] -= 1
                
                stim_i = ii - 1 
                tst_t_sum += 1
                #print('stim_i:',stim_i, 'tst_t_sum:',tst_t_sum)
                #print(stim_indptr_train)
                if tst_t_sum >= 50:
                    tst_t_sum -= 50 #print()
                    
                break
        
        
        #MUA_multi_train = MUA_multi[:, select_trial]
        
        mean_MUA_train = np.zeros([MUA_multi_train.shape[0], stim_indptr_train.shape[0]-1])
        for st in range(len(stim_indptr_train)-1):
            mean_MUA_train[:,st] = MUA_multi_train[:, stim_indptr_train[st]:stim_indptr_train[st+1]].mean(1)
            
        mean_MUA_mat_train = np.zeros(MUA_multi_train.shape)
        for st in range(len(stim_indptr_train)-1):
            mean_MUA_mat_train[:, stim_indptr_train[st]:stim_indptr_train[st+1]] = np.tile(mean_MUA_train[:,st].reshape(-1,1), stim_indptr_train[st+1]-stim_indptr_train[st])
        
        u, s, vh = np.linalg.svd((MUA_multi_train - mean_MUA_mat_train), full_matrices=False)
        approx = s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))
    
        #fluc_addi = approx[0,:]/approx[0,:].mean()
        fluc_addi = approx[0,:]/approx[0,:].std()
        print(fluc_addi)
        #print(approx[0,:])
        #print(approx[0,:].mean())
        #coup_addi = approx.mean(1)
        coup_addi = approx.std(1)
        
        for unit in range(len(record_neugroup)):
            #print('test_unit: ',unit)
            coup_addi_train = np.delete(coup_addi, unit)
            resid_MUA_unit_train = np.delete(MUA_multi[:,tst_t] - mean_MUA_train[:,stim_i], unit)
            fluc_addi_test = optimize.fmin(find_fluc, 0, (resid_MUA_unit_train, coup_addi_train),disp=True)
            #print(resid_MUA_unit_train, coup_addi_train)
            #print(fluc_addi_test)
            MUA_multi_predict[unit, tst_t] = mean_MUA_train[:,stim_i][unit] + fluc_addi_test*coup_addi[unit]

            
        # R = mydata.mydata()
        # R.fluc_addi = fluc_addi 
        # R.coup_addi = coup_addi
        # R.mean_MUA = mean_MUA
        # R.mean_MUA_mat =  mean_MUA_mat
        # R.MUA_multi = MUA_multi
        # R.u = u
        # R.s = s
        # R.vh = vh
        
    return MUA_multi, MUA_multi_predict
#%%
stim_indptr = np.arange(0, n_StimAmp*n_perStimAmp+1, n_perStimAmp)
#%%
MUA_multi_addi_noatt, MUA_multi_predict_addi_noatt = xvalid_addititive_fluc(data.a1.ge.spk_matrix, \
                                                     data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

    
#%%

def xvalid_indep(spk_mat, dura, record_neugroup, stim_indptr):
    
    MUA_multi = multiMUA_crosstrial(spk_mat, dura, record_neugroup)
    
    #select_trial = np.ones(MUA_multi.shape[1],dtype=bool)
    
    MUA_multi_train = MUA_multi[:,1:].copy()
    
    MUA_multi_predict = np.zeros(MUA_multi.shape)
    
    tst_t_sum = 0
    for tst_t in range(MUA_multi.shape[1]):
        #print('test_trial: ',tst_t)
        if tst_t > 0:
            MUA_multi_train[:,tst_t-1] = MUA_multi[:,tst_t-1]
            #if tst_t < 10:
                #print(MUA_multi_train[:,:10])
        # if tst_t == 0:
        #     select_trial[tst_t] = False
        # else:
        #     select_trial[tst_t] = False
        #     select_trial[tst_t-1] = True
        
        for ii in range(1,len(stim_indptr)):
            if tst_t < stim_indptr[ii]:
                stim_indptr_train = stim_indptr.copy()
                stim_indptr_train[ii:] -= 1
                
                stim_i = ii - 1 
                tst_t_sum += 1
                print('stim_i:',stim_i, 'tst_t_sum:',tst_t_sum)
                print(stim_indptr_train)
                if tst_t_sum >= 50:
                    tst_t_sum -= 50 #print()
                    
                break
        
        
        MUA_multi_predict[:,tst_t] = MUA_multi_train[:, stim_indptr_train[stim_i]:stim_indptr_train[stim_i+1]].mean(1)
        
    
    return MUA_multi, MUA_multi_predict


#%%
MUA_multi_indep_noatt, MUA_multi_predict_indep_noatt = xvalid_indep(data.a1.ge.spk_matrix, \
                                                     data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

#%%

def quality_index(MUA_multi_predict_test, MUA_multi_test, MUA_multi_predict_indep, MUA_multi_indep):
    e_test = np.sum(MUA_multi_predict_test + (MUA_multi_predict_test - MUA_multi_test)**2, 1)

    e_indep = np.sum(MUA_multi_predict_indep + (MUA_multi_predict_indep - MUA_multi_indep)**2, 1)
    
    return 1 - e_test/e_indep
#%%
q_addi = quality_index(MUA_multi_predict_addi_noatt, MUA_multi_addi_noatt, MUA_multi_predict_indep_noatt, MUA_multi_indep_noatt)
print(q_addi)
#%%
from scipy.stats import nbinom
#%%
mean, var = nbinom.stats(7, 1, moments='mv')
print(var, mean, var/mean)


#%%
solve_ambiguity = lambda k, mean_actvity, multip_coup, addi_coup: np.sum((multip_coup - k*addi_coup - mean_actvity)**2)

find_fluc_affine = lambda fluc,mc,ac,mean_activity: np.sum((mean_activity - (fluc[0]*mc+fluc[1]*ac))**2) # fluc[0]=m fluc[1]=a

#%%
import warnings
#%%

#%%
def xvalid_affine_fluc(spk_mat, dura, record_neugroup, stim_indptr):
    
    MUA_multi = multiMUA_crosstrial(spk_mat, dura, record_neugroup)
        
    MUA_multi_train = MUA_multi[:,1:].copy()
    
    MUA_multi_predict = np.zeros(MUA_multi.shape)
    
    tst_t_sum = 0
    for tst_t in range(MUA_multi.shape[1]):
        #print('test_trial: ',tst_t)
        if tst_t > 0:
            MUA_multi_train[:,tst_t-1] = MUA_multi[:,tst_t-1]
            #if tst_t < 10:
                #print(MUA_multi_train[:,:10])
        # if tst_t == 0:
        #     select_trial[tst_t] = False
        # else:
        #     select_trial[tst_t] = False
        #     select_trial[tst_t-1] = True
        
        for ii in range(1,len(stim_indptr)):
            if tst_t < stim_indptr[ii]:
                stim_indptr_train = stim_indptr.copy()
                stim_indptr_train[ii:] -= 1
                
                stim_i = ii - 1 
                tst_t_sum += 1
                #print('stim_i:',stim_i, 'tst_t_sum:',tst_t_sum)
                #print(stim_indptr_train)
                if tst_t_sum >= 50:
                    tst_t_sum -= 50 #print()
                    
                break

        approx_MUA_multip = _multiplicative_fluc(MUA_multi_train, stim_indptr_train)
        approx_MUA_addi = _additive_fluc(MUA_multi_train - approx_MUA_multip)
        approx_MUA_multip = _multiplicative_fluc(MUA_multi_train - approx_MUA_addi, stim_indptr_train)
        
        error_pre = ((MUA_multi_train - (approx_MUA_multip + approx_MUA_addi))**2).sum()/(MUA_multi_train.shape[0]*MUA_multi_train.shape[1])
        do_iteration = True
        #iter_n = 0
        while do_iteration:
            #print('iter_n:',iter_n); iter_n += 1
            approx_MUA_addi = _additive_fluc(MUA_multi_train - approx_MUA_multip)
            approx_MUA_multip = _multiplicative_fluc(MUA_multi_train - approx_MUA_addi, stim_indptr_train)
            error = ((MUA_multi_train - (approx_MUA_multip + approx_MUA_addi))**2).sum()/(MUA_multi_train.shape[0]*MUA_multi_train.shape[1])
            #print(error_pre - error)
            do_iteration = error_pre - error >= 1e-10
            error_pre = error
        
        #addi_coup_train = approx_MUA_addi.mean(1)
        addi_coup_train = approx_MUA_addi.std(1)
        
        multi_coup_1 = approx_MUA_multip[:,stim_indptr_train[0]:stim_indptr_train[1]].mean(1)
        MUA_mean_1 = MUA_multi_train[:,stim_indptr_train[0]:stim_indptr_train[1]].mean(1)
        
        k = optimize.fmin(solve_ambiguity, 0, (MUA_mean_1, multi_coup_1, addi_coup_train),disp=True)
        print('ambiguity:',k[0])
        
        multi_coup_train = approx_MUA_multip[:,stim_indptr_train[stim_i]:stim_indptr_train[stim_i+1]].mean(1)
        
        multi_coup_train = multi_coup_train - k[0]*addi_coup_train
        
        
        for unit in range(len(record_neugroup)):
            #print('test_unit: ',unit)
            addi_coup_train_unit = np.delete(addi_coup_train, unit)
            multi_coup_train_unit = np.delete(multi_coup_train, unit)
            MUA_unit_train = np.delete(MUA_multi[:,tst_t], unit)
            
            #find_fluc_affine = lambda m,a,mc,ac,mean_activity: np.sum((mean_activity - (m*mc+a*ac))**2)
            (fluc_m, fluc_a) = optimize.fmin(find_fluc_affine, [1,0], (multi_coup_train_unit, addi_coup_train_unit, MUA_unit_train),disp=True)
            print(fluc_m, fluc_a)
            MUA_multi_predict[unit, tst_t] = fluc_m*multi_coup_train[unit] + fluc_a*addi_coup_train[unit]
            #resid_MUA_unit_train = np.delete(MUA_multi[:,tst_t] - mean_MUA_train[:,stim_i], unit)
            #fluc_addi_test = optimize.fmin(find_fluc, 0, (resid_MUA_unit_train, coup_addi_train),disp=True)
            #print(resid_MUA_unit_train, coup_addi_train)
            #print(fluc_addi_test)
            #MUA_multi_predict[unit, tst_t] = mean_MUA_train[:,stim_i][unit] + fluc_addi_test*addi_coup_train[unit]
        
    return MUA_multi, MUA_multi_predict



#%%
#warnings.filterwarnings('error',category = Warning)#, message='*Maximum number of function*')

MUA_multi_affine_noatt, MUA_multi_predict_affine_noatt = xvalid_affine_fluc(data.a1.ge.spk_matrix, \
                                                     data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

#%%
#%%
q_addi = quality_index(MUA_multi_predict_affine_noatt, MUA_multi_affine_noatt, MUA_multi_predict_indep_noatt, MUA_multi_indep_noatt)
print(q_addi)

#%%
find_fluc_multip = lambda fluc,mc,mean_activity: np.sum((mean_activity - (fluc*mc))**2) # fluc[0]=m fluc[1]=a


#%%
def xvalid_multiplicative_fluc(spk_mat, dura, record_neugroup, stim_indptr):
    
    MUA_multi = multiMUA_crosstrial(spk_mat, dura, record_neugroup)
        
    MUA_multi_train = MUA_multi[:,1:].copy()
    
    MUA_multi_predict = np.zeros(MUA_multi.shape)
    
    tst_t_sum = 0
    for tst_t in range(MUA_multi.shape[1]):
        #print('test_trial: ',tst_t)
        if tst_t > 0:
            MUA_multi_train[:,tst_t-1] = MUA_multi[:,tst_t-1]
            #if tst_t < 10:
                #print(MUA_multi_train[:,:10])
        # if tst_t == 0:
        #     select_trial[tst_t] = False
        # else:
        #     select_trial[tst_t] = False
        #     select_trial[tst_t-1] = True
        
        for ii in range(1,len(stim_indptr)):
            if tst_t < stim_indptr[ii]:
                stim_indptr_train = stim_indptr.copy()
                stim_indptr_train[ii:] -= 1
                
                stim_i = ii - 1 
                tst_t_sum += 1
                #print('stim_i:',stim_i, 'tst_t_sum:',tst_t_sum)
                #print(stim_indptr_train)
                if tst_t_sum >= 50:
                    tst_t_sum -= 50 #print()
                    
                break

        approx_MUA_multip = _multiplicative_fluc(MUA_multi_train, stim_indptr_train)
        
    
        
        #addi_coup_train = approx_MUA_addi.mean(1)
        #addi_coup_train = approx_MUA_addi.std(1)
        
        # multi_coup_1 = approx_MUA_multip[:,stim_indptr_train[0]:stim_indptr_train[1]].mean(1)
        # MUA_mean_1 = MUA_multi_train[:,stim_indptr_train[0]:stim_indptr_train[1]].mean(1)
        
        # k = optimize.fmin(solve_ambiguity, 0, (MUA_mean_1, multi_coup_1, addi_coup_train),disp=True)
        # print('ambiguity:',k[0])
        
        multi_coup_train = approx_MUA_multip[:,stim_indptr_train[stim_i]:stim_indptr_train[stim_i+1]].mean(1)
        
        #multi_coup_train = multi_coup_train - k[0]*addi_coup_train
        
        
        for unit in range(len(record_neugroup)):
            #print('test_unit: ',unit)
            #addi_coup_train_unit = np.delete(addi_coup_train, unit)
            multi_coup_train_unit = np.delete(multi_coup_train, unit)
            MUA_unit_train = np.delete(MUA_multi[:,tst_t], unit)
            
            #find_fluc_affine = lambda m,a,mc,ac,mean_activity: np.sum((mean_activity - (m*mc+a*ac))**2)
            # (fluc_m, fluc_a) = optimize.fmin(find_fluc_affine, [1,0], (multi_coup_train_unit, addi_coup_train_unit, MUA_unit_train),disp=True)
            # print(fluc_m, fluc_a)
            find_fluc_multip = lambda fluc,mc,mean_activity: np.sum((mean_activity - (fluc*mc))**2) # fluc[0]=m fluc[1]=a
            fluc_m = optimize.fmin(find_fluc_multip, 1, (multi_coup_train_unit, MUA_unit_train),disp=True)
            print(fluc_m.shape)
            MUA_multi_predict[unit, tst_t] = fluc_m[0]*multi_coup_train[unit] #+ fluc_a*addi_coup_train[unit]
            #resid_MUA_unit_train = np.delete(MUA_multi[:,tst_t] - mean_MUA_train[:,stim_i], unit)
            #fluc_addi_test = optimize.fmin(find_fluc, 0, (resid_MUA_unit_train, coup_addi_train),disp=True)
            #print(resid_MUA_unit_train, coup_addi_train)
            #print(fluc_addi_test)
            #MUA_multi_predict[unit, tst_t] = mean_MUA_train[:,stim_i][unit] + fluc_addi_test*addi_coup_train[unit]
        
    return MUA_multi, MUA_multi_predict

#%%
#warnings.filterwarnings('error',category = Warning)#, message='*Maximum number of function*')

MUA_multi_mutp_noatt, MUA_multi_predict_mutp_noatt = xvalid_multiplicative_fluc(data.a1.ge.spk_matrix, \
                                                     data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

#%%
#%%
q_addi = quality_index(MUA_multi_predict_mutp_noatt, MUA_multi_mutp_noatt, MUA_multi_predict_indep_noatt, MUA_multi_indep_noatt)
print(q_addi)
#%%

plt.figure()
plt.plot(MUA_multi_mutp_noatt[:,51]) 
plt.plot(MUA_multi_predict_mutp_noatt[:,51])


#%%

def _multiplicative_fluc(spk_data, stim_indptr):
    
    approx_spk_data = np.zeros(spk_data.shape)
    
    for st in range(len(stim_indptr)-1):
        #spk_data[:,st] = spk_data[:, stim_indptr[st]:stim_indptr[st+1]].mean(1)

        u, s, vh = np.linalg.svd(spk_data[:, stim_indptr[st]:stim_indptr[st+1]], full_matrices=False)
        
        
        #approx_spk_data_tmp = s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))
        
        approx_spk_data[:, stim_indptr[st]:stim_indptr[st+1]] = s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))
        #fluc = approx_spk_data[0,:]/approx_spk_data[0,:].mean()
        
    return approx_spk_data
    
    
def _additive_fluc(residual_data):
    
    u, s, vh = np.linalg.svd(residual_data, full_matrices=False)
    #approx_residual_data = s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))

    # fluc_addi = approx[0,:]/approx[0,:].mean()
    # print(fluc_addi)
    # #print(approx[0,:])
    # #print(approx[0,:].mean())
    # coup_addi = approx.mean(1)   
    #return approx_residual_data
    return s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))


#%%
MUA_multi_mutp_noatt, MUA_multi_predict_mutp_noatt = xvalid_multiplicative_fluc(data.a1.ge.spk_matrix, \
                                                     data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)
#%%
MUA_multi_indep_noatt, MUA_multi_predict_indep_noatt = xvalid_indep(data.a1.ge.spk_matrix, \
                                                     data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

    
    #%%
#stim_indptr = np.array([0,50,100])
stim_indptr = np.array([0,50,100,150])

MUA_addi_noatt, MUA_predict_addi_noatt = gain_fluctuation.xvalid_addititive_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)


MUA_mutp_noatt, MUA_predict_mutp_noatt = gain_fluctuation.xvalid_multiplicative_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

MUA_affi_noatt, MUA_predict_affi_noatt = gain_fluctuation.xvalid_affine_fluc(data.a1.ge.spk_matrix, \
                                                 data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

#stim_indptr = np.array([0,50,100,])
#stim_indptr = np.array([0,50,100,150])

MUA_indep_noatt, MUA_predict_indep_noatt = gain_fluctuation.xvalid_indep(data.a1.ge.spk_matrix, \
                                            data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp].copy(), record_neugroup, stim_indptr)

#%%
q_mutp = quality_index(MUA_predict_mutp_noatt, MUA_mutp_noatt, MUA_predict_indep_noatt, MUA_indep_noatt)
print(q_mutp)
#%%
q_addi = quality_index(MUA_predict_addi_noatt, MUA_addi_noatt, MUA_predict_indep_noatt, MUA_indep_noatt)
print(q_addi)
#%%
q_affi = quality_index(MUA_predict_affi_noatt, MUA_affi_noatt, MUA_predict_indep_noatt, MUA_indep_noatt)
print(q_affi)

#%%

plt.figure()
for i in range(5):
    plt.plot(MUA_mutp_noatt[:,i],c=clr[0], marker='o')
    plt.plot(MUA_predict_mutp_noatt[:,i],c=clr[1])
    plt.plot(MUA_predict_indep_noatt[:,i],c=clr[1],ls='--')

#%%
plt.figure()
for i in range(0,1):
    plt.plot(MUA_addi_noatt[:,i],c=clr[0], marker='o')
    plt.plot(MUA_predict_addi_noatt[:,i],c=clr[1])
    #plt.plot(MUA_predict_indep_noatt[:,i],c=clr[1],ls='--')
#%%
plt.figure()
for i in range(5):
    plt.plot(MUA_affi_noatt[:,i],c=clr[0], marker='o')
    plt.plot(MUA_predict_affi_noatt[:,i],c=clr[1])
    plt.plot(MUA_predict_indep_noatt[:,i],c=clr[1],ls='--')
    
#%%
spon_dura_tmp = np.arange(5000,20001-250,250, dtype=int)
spon_dura = np.zeros([len(spon_dura_tmp),2], int)
spon_dura[:,0] = spon_dura_tmp
spon_dura[:,1] = spon_dura_tmp + 250
del spon_dura_tmp
dura_spon_st_noatt = np.vstack((spon_dura, data.a1.param.stim1.stim_on[:n_StimAmp*n_perStimAmp]))
dura_spon_st_att = np.vstack((spon_dura, data.a1.param.stim1.stim_on[n_StimAmp*n_perStimAmp:]))



#%%
#stim_indptr = np.array([0,50,100])
stim_indptr = np.hstack(([0],np.array([0,50,100,150]) + 60))

MUA_addi_sp_noatt, MUA_predict_addi_sp_noatt = gain_fluctuation.xvalid_addititive_fluc(data.a1.ge.spk_matrix, \
                           dura_spon_st_noatt.copy(), record_neugroup, stim_indptr)


MUA_mutp_sp_noatt, MUA_predict_mutp_sp_noatt = gain_fluctuation.xvalid_multiplicative_fluc(data.a1.ge.spk_matrix, \
                          dura_spon_st_noatt.copy(), record_neugroup, stim_indptr)

MUA_affi_sp_noatt, MUA_predict_affi_sp_noatt = gain_fluctuation.xvalid_affine_fluc(data.a1.ge.spk_matrix, \
                         dura_spon_st_noatt.copy(), record_neugroup, stim_indptr)

#stim_indptr = np.array([0,50,100,])
#stim_indptr = np.array([0,50,100,150])

MUA_indep_sp_noatt, MUA_predict_indep_sp_noatt = gain_fluctuation.xvalid_indep(data.a1.ge.spk_matrix, \
                          dura_spon_st_noatt.copy(), record_neugroup, stim_indptr)

#%%
q_mutp = quality_index(MUA_predict_mutp_sp_noatt, MUA_mutp_sp_noatt, MUA_predict_indep_sp_noatt, MUA_indep_sp_noatt)
print(q_mutp)

q_addi = quality_index(MUA_predict_addi_sp_noatt, MUA_addi_sp_noatt, MUA_predict_indep_sp_noatt, MUA_indep_sp_noatt)
print(q_addi)

q_affi = quality_index(MUA_predict_affi_sp_noatt, MUA_affi_sp_noatt, MUA_predict_indep_sp_noatt, MUA_indep_sp_noatt)
print(q_affi)

#%%

plt.figure()
for i in range(5):
    plt.plot(MUA_mutp_sp_noatt[:,i],c=clr[0], marker='o')
    plt.plot(MUA_predict_mutp_sp_noatt[:,i],c=clr[1])
    plt.plot(MUA_predict_indep_sp_noatt[:,i],c=clr[1],ls='--')

#%%
plt.figure()
for i in range(0,5):
    plt.plot(MUA_addi_sp_noatt[:,i],c=clr[0], marker='o')
    plt.plot(MUA_predict_addi_sp_noatt[:,i],c=clr[1])
    #plt.plot(MUA_predict_indep_noatt[:,i],c=clr[1],ls='--')
#%%
plt.figure()
for i in range(5):
    plt.plot(MUA_affi_noatt[:,i],c=clr[0], marker='o')
    plt.plot(MUA_predict_affi_noatt[:,i],c=clr[1])
    plt.plot(MUA_predict_indep_noatt[:,i],c=clr[1],ls='--')

