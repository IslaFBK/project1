#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:14:47 2021

@author: shni2598
"""


import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats
#%%
class fano_mean_match:
    
    def __init__(self):
        self.spk_sparmat = None # sparse matrix; time and neuron index of spikes
        self.stim_onoff = None # 2-D array; ms; on and off time of each stimulus # data.a1.param.stim.stim_on[20:40].copy() # ms stimulus onset, off       
        self.t_bf = 100 # ms; time before stimulus onset
        self.t_aft = 100 # ms; time after stimulus off
        self.dt = 0.1 # ms
        self.win = 150 # ms sliding window
        self.move_step = 10 # ms sliding window moving step
        self.bin_count_interval = 1 # number of spikes; interval of bin used to calculate the histogram of mean spike counts 
        self.repeat = 100 # repeat times in 'mean-matching'
        self.method = 'regression' # 'mean' or 'regression'
        self.mean_match_across_condition = False # if do mean matching across different condition e.g. attention or no-attention condition
        self.stim_onoff_2 = None #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True
        
    def get_fano(self):

        stim_onoff = np.round(self.stim_onoff/self.dt).astype(int) # ms stimulus onset, off
        t_bf = int(round(self.t_bf/self.dt)) # ms; time before stimulus onset
        t_aft = int(round(self.t_aft/self.dt)) # ms; time after stimulus off
        #dt = 0.1 # ms
        win = int(round(self.win/self.dt)) # ms sliding window
        hfwin = int(round(win/2))
        move_step = int(round(self.move_step/self.dt)) # ms sliding window moving step
        
        samp_onoff = np.copy(stim_onoff)
        samp_onoff[:,0] -= t_bf
        samp_onoff[:,1] += t_aft

        #stim_onoff = np.array([[2000,4000]])
        samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
        spk_count = np.zeros([self.spk_sparmat.shape[0],samp_stp.shape[0],samp_onoff.shape[0]])
        for i in range(samp_onoff.shape[0]):
            samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
            samp_stp += samp_onoff[i,0]
            for stp in range(len(samp_stp)):
                spk_count[:,stp,i] = self.spk_sparmat[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
        
        mean_var = np.zeros([self.spk_sparmat.shape[0],samp_stp.shape[0],2])
        
        mean_var[:,:,0] = spk_count.mean(2)
        mean_var[:,:,1] = spk_count.var(2)
        
        if self.mean_match_across_condition:
            stim_onoff = np.round(self.stim_onoff_2/self.dt).astype(int) # ms stimulus onset, off
            samp_onoff = np.copy(stim_onoff)
            samp_onoff[:,0] -= t_bf
            samp_onoff[:,1] += t_aft
    
            #stim_onoff = np.array([[2000,4000]])
            samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
            spk_count = np.zeros([self.spk_sparmat.shape[0],samp_stp.shape[0],samp_onoff.shape[0]])
            for i in range(samp_onoff.shape[0]):
                samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
                samp_stp += samp_onoff[i,0]
                for stp in range(len(samp_stp)):
                    spk_count[:,stp,i] = self.spk_sparmat[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
            
            mean_var_2 = np.zeros([self.spk_sparmat.shape[0],samp_stp.shape[0],2])
            
            mean_var_2[:,:,0] = spk_count.mean(2)
            mean_var_2[:,:,1] = spk_count.var(2)            
        
        
        
        #bin_count = np.arange(0.0001, spk_count.max()+1, self.bin_count_interval) 
        if not self.mean_match_across_condition:
            bin_count = np.arange(0.0001, mean_var[:,:,0].max()+1, self.bin_count_interval)# bins starts at 0.0001 to neglect zero-mean to avoid division by zero when calculating Fano
            #bin_count = np.arange(0, mean_var[:,:,0].max()+1, self.bin_count_interval) 
            mean_hist = np.zeros([bin_count.shape[0]-1, samp_stp.shape[0]], int)
            for t in range(samp_stp.shape[0]):
                hist, _ = np.histogram(mean_var[:,t,0],bin_count)
                mean_hist[:,t] = hist
            
            common_mean_hist = mean_hist.min(1)
        else:
            bin_count = np.arange(0.0001, max(mean_var[:,:,0].max(),mean_var_2[:,:,0].max())+1, self.bin_count_interval)# bins starts at 0.0001 to neglect zero-mean to avoid division by zero when calculating Fano
            #bin_count = np.arange(0, mean_var[:,:,0].max()+1, self.bin_count_interval) 
            mean_hist = np.zeros([bin_count.shape[0]-1, samp_stp.shape[0]], int)
            mean_hist_2 = np.zeros([bin_count.shape[0]-1, samp_stp.shape[0]], int)
            for t in range(samp_stp.shape[0]):
                hist, _ = np.histogram(mean_var[:,t,0],bin_count)
                mean_hist[:,t] = hist
                hist, _ = np.histogram(mean_var_2[:,t,0],bin_count)
                mean_hist_2[:,t] = hist
            
            common_mean_hist = np.concatenate((mean_hist, mean_hist_2),1).min(1)
        #samp_ratio = common_mean_hist.reshape(-1,1)/mean_hist
        
        #samp_ratio[np.isnan(samp_ratio)] = 0
        #bin_count_interval = bin_count[1] -bin_count[0]
        bin_non0 = bin_count[np.where(common_mean_hist != 0)[0]]
        common_mean_hist = common_mean_hist[common_mean_hist != 0]
        
        #repeat = 100
        fano_record = np.zeros([self.repeat, samp_stp.shape[0]])
        for t in range(samp_stp.shape[0]):
            #bin_non0 = bin_count[np.where(samp_ratio[:,t] != 0)[0]]
            #samp_ratio_non0 = samp_ratio[np.where(samp_ratio[:,t] != 0)[0]]
            for rpt in range(self.repeat):
                neu = [None]*len(bin_non0)
                for b in range(len(bin_non0)):
                    neu_chs = np.where((mean_var[:,t,0] >= bin_non0[b]) & (mean_var[:,t,0] < (bin_non0[b]+self.bin_count_interval)))[0]
                    neu_chs = np.random.choice(neu_chs, common_mean_hist[b],replace=False)
                    neu[b] = neu_chs
                neu = np.concatenate((neu))
                if self.method == 'mean':
                    fano = (mean_var[neu,t,1]/mean_var[neu,t,0]).mean()
                    fano_record[rpt, t] = fano
                elif self.method == 'regression':
                    model = LinearRegression()
                    model.fit_intercept = False
                    model.copy_X=False
                    model.fit(mean_var[neu,t,0].reshape(-1,1), mean_var[neu,t,1])
                    fano_record[rpt, t] = model.coef_[0]
        
        fano_mean = fano_record.mean(0)
        fano_std = fano_record.std(0)
        
        if self.mean_match_across_condition:
            fano_record_2 = np.zeros([self.repeat, samp_stp.shape[0]])
            for t in range(samp_stp.shape[0]):
                #bin_non0 = bin_count[np.where(samp_ratio[:,t] != 0)[0]]
                #samp_ratio_non0 = samp_ratio[np.where(samp_ratio[:,t] != 0)[0]]
                for rpt in range(self.repeat):
                    neu = [None]*len(bin_non0)
                    for b in range(len(bin_non0)):
                        neu_chs = np.where((mean_var_2[:,t,0] >= bin_non0[b]) & (mean_var_2[:,t,0] < (bin_non0[b]+self.bin_count_interval)))[0]
                        neu_chs = np.random.choice(neu_chs, common_mean_hist[b],replace=False)
                        neu[b] = neu_chs
                    neu = np.concatenate((neu))
                    if self.method == 'mean':
                        fano = (mean_var_2[neu,t,1]/mean_var_2[neu,t,0]).mean()
                        fano_record_2[rpt, t] = fano
                    elif self.method == 'regression':
                        model = LinearRegression()
                        model.fit_intercept = False
                        model.copy_X=False
                        model.fit(mean_var_2[neu,t,0].reshape(-1,1), mean_var_2[neu,t,1])
                        fano_record_2[rpt, t] = model.coef_[0]
            
            fano_mean_2 = fano_record_2.mean(0)
            fano_std_2 = fano_record_2.std(0)
        
        
        if not self.mean_match_across_condition:
            return fano_mean, fano_std, fano_record
        else:
            return fano_mean, fano_std, fano_record, fano_mean_2, fano_std_2, fano_record_2

#%%
class fano_mean_match_2:
    
    def __init__(self):
        self.spk_sparmat = None # sparse matrix; time and neuron index of spikes
        self.stim_onoff = None # 2-D array; ms; on and off time of each stimulus # data.a1.param.stim.stim_on[20:40].copy() # ms stimulus onset, off       
        self.t_bf = 100 # ms; time before stimulus onset
        self.t_aft = 100 # ms; time after stimulus off
        self.dt = 0.1 # ms
        self.win = 150 # ms sliding window
        self.move_step = 10 # ms sliding window moving step
        self.bin_count_interval = 1 # number of spikes; interval of bin used to calculate the histogram of mean spike counts 
        self.repeat = 100 # repeat times in 'mean-matching'
        self.method = 'regression' # 'mean' or 'regression'
        self.mean_match_across_condition = False # if do mean matching across different condition e.g. attention or no-attention condition
        self.stim_onoff_2 = None #  on and off time of each stimulus under condition 2; this variable takes effect only when self.mean_match_across_condition =  True
        self.seed = 100
        
    def get_fano(self):

        stim_onoff = np.round(self.stim_onoff/self.dt).astype(int) # ms stimulus onset, off
        t_bf = int(round(self.t_bf/self.dt)) # ms; time before stimulus onset
        t_aft = int(round(self.t_aft/self.dt)) # ms; time after stimulus off
        #dt = 0.1 # ms
        win = int(round(self.win/self.dt)) # ms sliding window
        hfwin = int(round(win/2))
        move_step = int(round(self.move_step/self.dt)) # ms sliding window moving step
        
        samp_onoff = np.copy(stim_onoff)
        samp_onoff[:,0] -= t_bf
        samp_onoff[:,1] += t_aft
        
        samp_stp = np.arange(0, samp_onoff[0,1] - samp_onoff[0,0] + 1, move_step)
        #stim_onoff = np.array([[2000,4000]])
        mean_var, spk_count = get_mean_var_spk(self.spk_sparmat, samp_onoff, hfwin, move_step)
        '''
        samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
        spk_count = np.zeros([self.spk_sparmat.shape[0],samp_stp.shape[0],samp_onoff.shape[0]])
        for i in range(samp_onoff.shape[0]):
            samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
            samp_stp += samp_onoff[i,0]
            for stp in range(len(samp_stp)):
                spk_count[:,stp,i] = self.spk_sparmat[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
        
        mean_var = np.zeros([self.spk_sparmat.shape[0],samp_stp.shape[0],2])
        
        mean_var[:,:,0] = spk_count.mean(2)
        mean_var[:,:,1] = spk_count.var(2)
        '''
        if self.mean_match_across_condition:
            stim_onoff = np.round(self.stim_onoff_2/self.dt).astype(int) # ms stimulus onset, off
            samp_onoff = np.copy(stim_onoff)
            samp_onoff[:,0] -= t_bf
            samp_onoff[:,1] += t_aft
            
            samp_stp_2 = np.arange(0, samp_onoff[0,1] - samp_onoff[0,0] + 1, move_step)
            #stim_onoff = np.array([[2000,4000]])
            mean_var_2, spk_count_2 = get_mean_var_spk(self.spk_sparmat, samp_onoff, hfwin, move_step)
            '''
            samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
            spk_count = np.zeros([self.spk_sparmat.shape[0],samp_stp.shape[0],samp_onoff.shape[0]])
            for i in range(samp_onoff.shape[0]):
                samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
                samp_stp += samp_onoff[i,0]
                for stp in range(len(samp_stp)):
                    spk_count[:,stp,i] = self.spk_sparmat[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
            
            mean_var_2 = np.zeros([self.spk_sparmat.shape[0],samp_stp.shape[0],2])
            
            mean_var_2[:,:,0] = spk_count.mean(2)
            mean_var_2[:,:,1] = spk_count.var(2)            
            '''
        
        
        #bin_count = np.arange(0.0001, spk_count.max()+1, self.bin_count_interval) 
        
        if not self.mean_match_across_condition:
            bin_count = np.arange(0.0001, mean_var[:,:,0].max()+1, self.bin_count_interval)# bins starts at 0.0001 to neglect zero-mean to avoid division by zero when calculating Fano
            #bin_count = np.arange(0, mean_var[:,:,0].max()+1, self.bin_count_interval) 
            mean_hist = np.zeros([bin_count.shape[0]-1, samp_stp.shape[0]], int)
            for t in range(samp_stp.shape[0]):
                hist, _ = np.histogram(mean_var[:,t,0],bin_count)
                mean_hist[:,t] = hist
            
            common_mean_hist = mean_hist.min(1)
        else:
            bin_count = np.arange(0.0001, max(mean_var[:,:,0].max(),mean_var_2[:,:,0].max())+1, self.bin_count_interval)# bins starts at 0.0001 to neglect zero-mean to avoid division by zero when calculating Fano
            #bin_count = np.arange(0, mean_var[:,:,0].max()+1, self.bin_count_interval) 
            mean_hist = np.zeros([bin_count.shape[0]-1, samp_stp.shape[0]], int)
            mean_hist_2 = np.zeros([bin_count.shape[0]-1, samp_stp_2.shape[0]], int)
            for t in range(samp_stp.shape[0]):
                hist, _ = np.histogram(mean_var[:,t,0],bin_count)
                mean_hist[:,t] = hist
            for t in range(samp_stp_2.shape[0]):
                hist, _ = np.histogram(mean_var_2[:,t,0],bin_count)
                mean_hist_2[:,t] = hist
            
            common_mean_hist = np.concatenate((mean_hist, mean_hist_2),1).min(1)
        common_mean_hist_all = common_mean_hist.copy()
        #samp_ratio = common_mean_hist.reshape(-1,1)/mean_hist
        
        #samp_ratio[np.isnan(samp_ratio)] = 0
        #bin_count_interval = bin_count[1] -bin_count[0]
        bin_non0 = bin_count[np.where(common_mean_hist != 0)[0]]
        common_mean_hist = common_mean_hist[common_mean_hist != 0]
        
        #repeat = 100
        np.random.seed(self.seed)
        fano_mean, fano_sem, fano_record, mean_record = sampling_fano(mean_var, bin_non0, common_mean_hist, self.bin_count_interval, self.repeat, samp_stp, method = 'regression')
        '''
        fano_record = np.zeros([self.repeat, samp_stp.shape[0]])
        for t in range(samp_stp.shape[0]):
            #bin_non0 = bin_count[np.where(samp_ratio[:,t] != 0)[0]]
            #samp_ratio_non0 = samp_ratio[np.where(samp_ratio[:,t] != 0)[0]]
            for rpt in range(self.repeat):
                neu = [None]*len(bin_non0)
                for b in range(len(bin_non0)):
                    neu_chs = np.where((mean_var[:,t,0] >= bin_non0[b]) & (mean_var[:,t,0] < (bin_non0[b]+self.bin_count_interval)))[0]
                    neu_chs = np.random.choice(neu_chs, common_mean_hist[b],replace=False)
                    neu[b] = neu_chs
                neu = np.concatenate((neu))
                if self.method == 'mean':
                    fano = (mean_var[neu,t,1]/mean_var[neu,t,0]).mean()
                    fano_record[rpt, t] = fano
                elif self.method == 'regression':
                    model = LinearRegression()
                    model.fit_intercept = False
                    model.copy_X=False
                    model.fit(mean_var[neu,t,0].reshape(-1,1), mean_var[neu,t,1])
                    fano_record[rpt, t] = model.coef_[0]
        
        fano_mean = fano_record.mean(0)
        fano_std = fano_record.std(0)
        '''
        if self.mean_match_across_condition:
            
            fano_mean_2, fano_sem_2, fano_record_2, mean_record_2 = sampling_fano(mean_var_2, bin_non0, common_mean_hist, self.bin_count_interval, self.repeat, samp_stp_2, method = 'regression')
            '''
            fano_record_2 = np.zeros([self.repeat, samp_stp.shape[0]])
            for t in range(samp_stp.shape[0]):
                #bin_non0 = bin_count[np.where(samp_ratio[:,t] != 0)[0]]
                #samp_ratio_non0 = samp_ratio[np.where(samp_ratio[:,t] != 0)[0]]
                for rpt in range(self.repeat):
                    neu = [None]*len(bin_non0)
                    for b in range(len(bin_non0)):
                        neu_chs = np.where((mean_var_2[:,t,0] >= bin_non0[b]) & (mean_var_2[:,t,0] < (bin_non0[b]+self.bin_count_interval)))[0]
                        neu_chs = np.random.choice(neu_chs, common_mean_hist[b],replace=False)
                        neu[b] = neu_chs
                    neu = np.concatenate((neu))
                    if self.method == 'mean':
                        fano = (mean_var_2[neu,t,1]/mean_var_2[neu,t,0]).mean()
                        fano_record_2[rpt, t] = fano
                    elif self.method == 'regression':
                        model = LinearRegression()
                        model.fit_intercept = False
                        model.copy_X=False
                        model.fit(mean_var_2[neu,t,0].reshape(-1,1), mean_var_2[neu,t,1])
                        fano_record_2[rpt, t] = model.coef_[0]
            
            fano_mean_2 = fano_record_2.mean(0)
            fano_std_2 = fano_record_2.std(0)
            '''
        
        if not self.mean_match_across_condition:
            return fano_mean, fano_sem, fano_record, mean_record, common_mean_hist_all
        else:
            return fano_mean, fano_sem, fano_record, mean_record, fano_mean_2, fano_sem_2, fano_record_2, mean_record_2, common_mean_hist_all

def get_mean_var_spk(spk_sparmat, samp_onoff, hfwin, move_step):
        # stim_onoff = np.round(self.stim_onoff_2/self.dt).astype(int) # ms stimulus onset, off
        # samp_onoff = np.copy(stim_onoff)
        # samp_onoff[:,0] -= t_bf
        # samp_onoff[:,1] += t_aft

        #stim_onoff = np.array([[2000,4000]])
        samp_stp = np.arange(0, samp_onoff[0,1] - samp_onoff[0,0] + 1, move_step)
        spk_count = np.zeros([spk_sparmat.shape[0],samp_stp.shape[0],samp_onoff.shape[0]])
        for i in range(samp_onoff.shape[0]):
            samp_stp = np.arange(0, samp_onoff[0,1] - samp_onoff[0,0] + 1, move_step)
            samp_stp += samp_onoff[i,0]
            for stp in range(len(samp_stp)):
                spk_count[:,stp,i] = spk_sparmat[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
        
        mean_var = np.zeros([spk_sparmat.shape[0],samp_stp.shape[0],2])
        
        mean_var[:,:,0] = spk_count.mean(2)
        mean_var[:,:,1] = spk_count.var(2)  
        
        return mean_var, spk_count

def sampling_fano(mean_var, bin_non0, common_mean_hist, bin_count_interval, repeat, samp_stp, method = 'regression'):
    
    fano_record = np.zeros([repeat, samp_stp.shape[0]])
    mean_record = np.zeros([repeat, samp_stp.shape[0]])
    for t in range(samp_stp.shape[0]):
        #bin_non0 = bin_count[np.where(samp_ratio[:,t] != 0)[0]]
        #samp_ratio_non0 = samp_ratio[np.where(samp_ratio[:,t] != 0)[0]]
        for rpt in range(repeat):
            neu = [None]*len(bin_non0)
            for b in range(len(bin_non0)):
                neu_chs = np.where((mean_var[:,t,0] >= bin_non0[b]) & (mean_var[:,t,0] < (bin_non0[b]+bin_count_interval)))[0]
                neu_chs = np.random.choice(neu_chs, common_mean_hist[b],replace=False)
                neu[b] = neu_chs
            neu = np.concatenate((neu))
            if method == 'mean':
                fano = (mean_var[neu,t,1]/mean_var[neu,t,0]).mean()
                fano_record[rpt, t] = fano
                mean_record[rpt, t] = mean_var[neu,t,0].mean()
            elif method == 'regression':
                model = LinearRegression()
                model.fit_intercept = False
                model.copy_X=False
                model.fit(mean_var[neu,t,0].reshape(-1,1), mean_var[neu,t,1])
                fano_record[rpt, t] = model.coef_[0]
                mean_record[rpt, t] = mean_var[neu,t,0].mean()
                
    fano_mean = fano_record.mean(0)
    fano_sem = scipy.stats.sem(fano_record, 0, nan_policy='omit')
    #fano_std= fano_record.std(0)
    
    return fano_mean, fano_sem, fano_record, mean_record
    
    
# def makelattice(n, width, centre):
    
# #    width =49
# #    n = 50 # number of neurons on each side of the grid
#     # hw = width/2
    
#     # x = np.linspace(-hw,hw,n) + centre[0]
#     # y = np.linspace(hw,-hw,n) + centre[1]
    
#     hw = width/2 #31.5
#     step = width/n
#     x = np.linspace(-hw + step/2, hw - step/2, n) + centre[0]
#     y = np.linspace(hw - step/2, -hw + step/2, n) + centre[1]
    
#     xx, yy = np.meshgrid(x,y)
    
#     lattice = np.zeros([n*n, 2])
#     lattice[:,0] = xx.flatten()
#     lattice[:,1] = yy.flatten()
    
#     return lattice
# #%%


# stim_loc = np.array([0,0])

# neuron = np.arange(4096)

# e_lattice = makelattice(64, 64, [0,0])

# dist = cn.coordination.lattice_dist(e_lattice,64,stim_loc)
# neu_pool = [None]*1
# neu_pool[0] = neuron[(dist >= 0) & (dist <= 10)]