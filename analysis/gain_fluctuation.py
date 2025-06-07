#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:03:10 2021

@author: shni2598
"""


import mydata
import numpy as np
from scipy import optimize


#%%

find_fluc_addi = lambda fluc, ac, train_activity: np.sum((train_activity - fluc*ac)**2)

find_fluc_addi_exact = lambda ac, train_activity: (ac*train_activity).sum()/(ac*ac).sum()

#%%    
# cp = np.array([2,1,3,4,5])
# train = np.array([1,2,5,6,4])
# fluc_addi_test = optimize.fmin(find_fluc_addi, 0, (cp, train),disp=False)

# find_fluc_addi_exact(cp, train)

#%%
find_fluc_multip = lambda fluc,mc,train_activity: np.sum((train_activity - (fluc*mc))**2) # fluc[0]=m fluc[1]=a

find_fluc_multip_exact = lambda mc, train_activity: (mc*train_activity).sum()/(mc*mc).sum()

#%%
solve_ambiguity = lambda k, mean_actvity, multip_coup, addi_coup: np.sum((multip_coup - k*addi_coup - mean_actvity)**2)

solve_ambiguity_exact = lambda mean_actvity, mc, ac: (ac*(mc-mean_actvity)).sum()/(ac*ac).sum()
#%%
# mc = np.array([2,3,2,4,5])
# ac = np.array([2,1,3,4,5])
# train = np.array([1,2,5,6,4])*2

# k = optimize.fmin(solve_ambiguity, 0, (train, mc, ac),disp=False)
# solve_ambiguity_exact(train, mc, ac)
#%%
find_fluc_affine = lambda fluc,mc,ac,train_activity: np.sum((train_activity - (fluc[0]*mc+fluc[1]*ac))**2) # fluc[0]=m fluc[1]=a

def find_fluc_affine_exact(mc, ac, train_activity):
    coup = np.hstack((mc.reshape(-1,1),ac.reshape(-1,1)))
    return np.dot(np.dot(np.linalg.inv(np.dot(coup.T,coup)), coup.T), train_activity.reshape(-1,1)).reshape(-1)
#%%
# mc = np.array([2,3,2,4,5])
# ac = np.array([2,1,3,4,5])
# train = np.array([1,2,5,6,4])*2

# (fluc_m, fluc_a) = find_fluc_affine_exact(mc, ac, train)
# #%%
# (fluc_m, fluc_a) = optimize.fmin(find_fluc_affine, [1,0], (mc, ac, train),disp=False)

#%%
def quality_index(MUA_multi_predict_test, MUA_multi_test, MUA_multi_predict_indep, MUA_multi_indep):
    e_test = np.sum(MUA_multi_predict_test + (MUA_multi_predict_test - MUA_multi_test)**2, 1)

    e_indep = np.sum(MUA_multi_predict_indep + (MUA_multi_predict_indep - MUA_multi_indep)**2, 1)
    
    return 1 - e_test/e_indep

#%%
def _multiplicative_fluc(spk_data, stim_indptr):
    
    approx_spk_data = np.zeros(spk_data.shape)
    
    for st in range(len(stim_indptr)-1):
        #spk_data[:,st] = spk_data[:, stim_indptr[st]:stim_indptr[st+1]].mean(1)

        u, s, vh = np.linalg.svd(spk_data[:, stim_indptr[st]:stim_indptr[st+1]], full_matrices=False)
        
        approx_spk_data[:, stim_indptr[st]:stim_indptr[st+1]] = s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))
        
    return approx_spk_data
    
    
def _additive_fluc(residual_data):
    
    u, s, vh = np.linalg.svd(residual_data, full_matrices=False)

    return s[0]*np.dot(u[:,0].reshape(-1,1), vh[0,:].reshape(1,-1))


def multiMUA_crosstrial(spk_mat, dura, record_neugroup, dt=0.1):
    
    dt_ = int(round(1/dt))
    spk_mat = spk_mat.tocsr()
    
    MUA = np.zeros([len(record_neugroup), len(dura)])
    for neu_i, neu in enumerate(record_neugroup):
        for stim_i, stim_dura in enumerate(dura):
            MUA[neu_i, stim_i] = spk_mat[neu, stim_dura[0]*dt_:stim_dura[1]*dt_].sum()
            
    return MUA

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
    R.stim_indptr = stim_indptr
    #R.u = u
    #R.s = s
    #R.vh = vh
    
    return R
#%%
def get_multiplicative_fluc(spk_mat, dura, record_neugroup, stim_indptr):
    
    MUA_multi = multiMUA_crosstrial(spk_mat, dura, record_neugroup)
    
    approx_MUA_multip = _multiplicative_fluc(MUA_multi, stim_indptr)
    
    fluc_multip = np.zeros(approx_MUA_multip.shape[1])
    coup_multip = np.zeros([approx_MUA_multip.shape[0], len(stim_indptr)-1])
    for st in range(len(stim_indptr)-1):
        fluc_multip[stim_indptr[st]:stim_indptr[st+1]] = approx_MUA_multip[0,stim_indptr[st]:stim_indptr[st+1]]/approx_MUA_multip[0,stim_indptr[st]:stim_indptr[st+1]].mean()
        coup_multip[:,st] = approx_MUA_multip[:,stim_indptr[st]:stim_indptr[st+1]].mean(1)
    
    R = mydata.mydata()
    R.fluc_multip = fluc_multip 
    R.coup_multip = coup_multip
    R.approx_MUA_multip = approx_MUA_multip
    R.MUA_multi =  MUA_multi
    R.stim_indptr = stim_indptr

    return R
    
#%%
def get_affine_fluc(spk_mat, dura, record_neugroup, stim_indptr, trialforSolveAmbiguity=0):
    
    MUA_multi = multiMUA_crosstrial(spk_mat, dura, record_neugroup)
    

    approx_MUA_multip = _multiplicative_fluc(MUA_multi, stim_indptr)
    approx_MUA_addi = _additive_fluc(MUA_multi - approx_MUA_multip)
    approx_MUA_multip = _multiplicative_fluc(MUA_multi - approx_MUA_addi, stim_indptr)
    
    error_pre = ((MUA_multi - (approx_MUA_multip + approx_MUA_addi))**2).sum()/(MUA_multi.shape[0]*MUA_multi.shape[1])
    do_iteration = True
    #iter_n = 0
    while do_iteration:
        #print('iter_n:',iter_n); iter_n += 1
        approx_MUA_addi = _additive_fluc(MUA_multi - approx_MUA_multip)
        approx_MUA_multip = _multiplicative_fluc(MUA_multi - approx_MUA_addi, stim_indptr)
        error = ((MUA_multi - (approx_MUA_multip + approx_MUA_addi))**2).sum()/(MUA_multi.shape[0]*MUA_multi.shape[1])
        #print(error_pre - error)
        do_iteration = error_pre - error >= 1e-10
        error_pre = error
    
    #addi_coup_train = approx_MUA_addi.mean(1)
    coup_addi = approx_MUA_addi.std(1)
    fluc_addi = approx_MUA_addi[0,:]/coup_addi[0]
    #print((approx_MUA_addi[0,:]/coup_addi[0])[0:2], (approx_MUA_addi[1,:]/coup_addi[1])[0:2])


    fluc_multip = np.zeros(approx_MUA_multip.shape[1])
    coup_multip = np.zeros([approx_MUA_multip.shape[0], len(stim_indptr)-1])
    for st in range(len(stim_indptr)-1):
        fluc_multip[stim_indptr[st]:stim_indptr[st+1]] = approx_MUA_multip[0,stim_indptr[st]:stim_indptr[st+1]]/approx_MUA_multip[0,stim_indptr[st]:stim_indptr[st+1]].mean()
        coup_multip[:,st] = approx_MUA_multip[:,stim_indptr[st]:stim_indptr[st+1]].mean(1)
    
    
    
    multi_coup_1 = coup_multip[:, trialforSolveAmbiguity] # approx_MUA_multip[:,stim_indptr[0]:stim_indptr[1]].mean(1)
    MUA_mean_1 = MUA_multi[:,stim_indptr[trialforSolveAmbiguity]:stim_indptr[trialforSolveAmbiguity+1]].mean(1)
    
    #k = optimize.fmin(solve_ambiguity, 0, (MUA_mean_1, multi_coup_1, coup_addi),disp=False)
    k = solve_ambiguity_exact(MUA_mean_1, multi_coup_1, coup_addi)
    print('ambiguity:',k)
    
    #coup_multip -= k[0]*coup_addi.reshape(-1,1)
    #fluc_addi += k[0]*fluc_multip
    coup_multip -= k*coup_addi.reshape(-1,1)
    fluc_addi += k*fluc_multip

    
    # multi_coup_train = approx_MUA_multip[:,stim_indptr[stim_i]:stim_indptr[stim_i+1]].mean(1)
    
    # multi_coup_train = multi_coup_train - k[0]*addi_coup_train
    R = mydata.mydata()
    R.fluc_multip = fluc_multip 
    R.coup_multip = coup_multip
    R.fluc_addi = fluc_addi 
    R.coup_addi = coup_addi    
    
    R.approx_MUA_multip = approx_MUA_multip
    R.approx_MUA_addi = approx_MUA_addi
    R.MUA_multi =  MUA_multi
    R.stim_indptr = stim_indptr
        
    return R
    
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
                #print('stim_i:',stim_i, 'tst_t_sum:',tst_t_sum)
                #print(stim_indptr_train)
                if tst_t_sum >= 50:
                    tst_t_sum -= 50 #print()
                    
                break
        
        
        MUA_multi_predict[:,tst_t] = MUA_multi_train[:, stim_indptr_train[stim_i]:stim_indptr_train[stim_i+1]].mean(1)
        
    
    return MUA_multi, MUA_multi_predict



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
        #fluc_addi = approx[0,:]/approx[0,:].std()
        #print(fluc_addi)
        #print(approx[0,:])
        #print(approx[0,:].mean())
        #coup_addi = approx.mean(1)
        coup_addi = approx.std(1)
        
        for unit in range(len(record_neugroup)):
            #print('test_unit: ',unit)
            coup_addi_train = np.delete(coup_addi, unit)
            resid_MUA_unit_train = np.delete(MUA_multi[:,tst_t] - mean_MUA_train[:,stim_i], unit)
            
            #fluc_addi_test = optimize.fmin(find_fluc_addi, 0, (coup_addi_train, resid_MUA_unit_train),disp=False)
            fluc_addi_test = find_fluc_addi_exact(coup_addi_train, resid_MUA_unit_train)
            
            #print(resid_MUA_unit_train, coup_addi_train)
            #print(fluc_addi_test)
            MUA_multi_predict[unit, tst_t] = mean_MUA_train[:,stim_i][unit] + fluc_addi_test*coup_addi[unit]

        
    return MUA_multi, MUA_multi_predict



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
        
        
        multi_coup_train = approx_MUA_multip[:,stim_indptr_train[stim_i]:stim_indptr_train[stim_i+1]].mean(1)
        
        for unit in range(len(record_neugroup)):
            multi_coup_train_unit = np.delete(multi_coup_train, unit)
            MUA_unit_train = np.delete(MUA_multi[:,tst_t], unit)
            
            #fluc_m = optimize.fmin(find_fluc_multip, 1, (multi_coup_train_unit, MUA_unit_train),disp=False)
            fluc_m = find_fluc_multip_exact(multi_coup_train_unit, MUA_unit_train)
            #print(fluc_m.shape)
            #MUA_multi_predict[unit, tst_t] = fluc_m[0]*multi_coup_train[unit] #+ fluc_a*addi_coup_train[unit]
            MUA_multi_predict[unit, tst_t] = fluc_m*multi_coup_train[unit] #+ fluc_a*addi_coup_train[unit]
        
    return MUA_multi, MUA_multi_predict



#%%
def xvalid_affine_fluc(spk_mat, dura, record_neugroup, stim_indptr, trialforSolveAmbiguity=0):
    
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
        
        tsa = trialforSolveAmbiguity
        multi_coup_1 = approx_MUA_multip[:,stim_indptr_train[tsa]:stim_indptr_train[tsa+1]].mean(1)
        MUA_mean_1 = MUA_multi_train[:,stim_indptr_train[tsa]:stim_indptr_train[tsa+1]].mean(1)
        
        #k = optimize.fmin(solve_ambiguity, 0, (MUA_mean_1, multi_coup_1, addi_coup_train),disp=False)
        k = solve_ambiguity_exact(MUA_mean_1, multi_coup_1, addi_coup_train)

        print('ambiguity:',k)
        
        multi_coup_train = approx_MUA_multip[:,stim_indptr_train[stim_i]:stim_indptr_train[stim_i+1]].mean(1)
        
        #multi_coup_train = multi_coup_train - k[0]*addi_coup_train
        multi_coup_train = multi_coup_train - k*addi_coup_train
        
        
        for unit in range(len(record_neugroup)):
            #print('test_unit: ',unit)
            addi_coup_train_unit = np.delete(addi_coup_train, unit)
            multi_coup_train_unit = np.delete(multi_coup_train, unit)
            MUA_unit_train = np.delete(MUA_multi[:,tst_t], unit)
            
            #find_fluc_affine = lambda m,a,mc,ac,mean_activity: np.sum((mean_activity - (m*mc+a*ac))**2)
            if np.all(multi_coup_train_unit/addi_coup_train_unit == (multi_coup_train_unit/addi_coup_train_unit)[0]):                
                (fluc_m, fluc_a) = optimize.fmin(find_fluc_affine, [1,0], (multi_coup_train_unit, addi_coup_train_unit, MUA_unit_train),disp=False)
            else:
                (fluc_m, fluc_a) = find_fluc_affine_exact(multi_coup_train_unit, addi_coup_train_unit, MUA_unit_train)
            
            #print(fluc_m, fluc_a)
            MUA_multi_predict[unit, tst_t] = fluc_m*multi_coup_train[unit] + fluc_a*addi_coup_train[unit]
            #resid_MUA_unit_train = np.delete(MUA_multi[:,tst_t] - mean_MUA_train[:,stim_i], unit)
            #fluc_addi_test = optimize.fmin(find_fluc, 0, (resid_MUA_unit_train, coup_addi_train),disp=True)
            #print(resid_MUA_unit_train, coup_addi_train)
            #print(fluc_addi_test)
            #MUA_multi_predict[unit, tst_t] = mean_MUA_train[:,stim_i][unit] + fluc_addi_test*addi_coup_train[unit]
        
    return MUA_multi, MUA_multi_predict
