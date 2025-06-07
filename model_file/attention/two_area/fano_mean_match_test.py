#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:43:24 2021

@author: shni2598
"""
import numpy as np
import fano_mean_match
#%%
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

spk_respon = np.zeros([neu_pool[0].shape[0], N_stim])

for i in range(N_stim):
    spk_respon[:,i] = data.a1.ge.spk_matrix[neu_pool[0], data.a1.param.stim.stim_on[i,0]*10:(data.a1.param.stim.stim_on[i,0]+fr_bin[0])*10].sum(1).A[:,0]

#%%
analy_type = 'var'
#datapath = ''
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/var/'
sys_argv = 0#int(sys.argv[1])
loop_num = sys_argv #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
#%%
stim_loc = np.array([0,0])

neuron = np.arange(data.a1.param.Ne)

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,63,stim_loc)
neu_pool = [None]*1
neu_pool[0] = neuron[(dist >= 0) & (dist <= 10)]
#%%
data.a1.ge.spk_matrix[neu_pool[0],:]

#%%
#data.a1.param.stim.stim_on[i,0]

spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]

#spk_count = 0
stim_onoff = data.a1.param.stim.stim_on[20:40].copy() # ms stimulus onset, off
t_bf = 100 # ms; time before stimulus onset
t_aft = 100 # ms; time after stimulus off
dt = 0.1 # ms
win = 150 # ms sliding window
move_step = 10 # ms sliding window moving step


stim_onoff = np.round(stim_onoff/dt).astype(int) # ms stimulus onset, off
t_bf = int(round(t_bf/dt)) # ms; time before stimulus onset
t_aft = int(round(t_aft/dt)) # ms; time after stimulus off
dt = 0.1 # ms
win = int(round(win/dt)) # ms sliding window
hfwin = int(round(win/2))
move_step = int(round(move_step/dt)) # ms sliding window moving step

samp_onoff = np.copy(stim_onoff)
samp_onoff[:,0] -= t_bf
samp_onoff[:,1] += t_aft

#stim_onoff = np.array([[2000,4000]])
samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
spk_count = np.zeros([spk_sparmat.shape[0],samp_stp.shape[0],samp_onoff.shape[0]])
for i in range(samp_onoff.shape[0]):
    samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
    samp_stp += samp_onoff[i,0]
    for stp in range(len(samp_stp)):
        spk_count[:,stp,i] = spk_sparmat[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)

mean_var = np.zeros([spk_sparmat.shape[0],samp_stp.shape[0],2])

mean_var[:,:,0] = spk_count.mean(2)
mean_var[:,:,1] = spk_count.var(2)
#%%
bin_count = np.arange(0.0001, mean_var[:,:,0].max()+1, 0.3) 
mean_hist = np.zeros([bin_count.shape[0]-1, samp_stp.shape[0]], int)
for t in range(samp_stp.shape[0]):
    hist, _ = np.histogram(mean_var[:,t,0],bin_count)
    mean_hist[:,t] = hist

common_mean_hist = mean_hist.min(1)
#samp_ratio = common_mean_hist.reshape(-1,1)/mean_hist
#%%
#samp_ratio[np.isnan(samp_ratio)] = 0
bin_count_interval = bin_count[1] -bin_count[0]
bin_non0 = bin_count[np.where(common_mean_hist != 0)[0]]
common_mean_hist = common_mean_hist[common_mean_hist != 0]

repeat = 100
fano_record = np.zeros([repeat, samp_stp.shape[0]])
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
        fano = (mean_var[neu,t,1]/mean_var[neu,t,0]).mean()
        fano_record[rpt, t] = fano

fano_mean = fano_record.mean(0)
fano_std = fano_record.std(0)
#%%
fig, ax = plt.subplots(1,1)
ax.errorbar(np.arange(fano_mean.shape[0])*10-100,fano_mean,fano_std)

#%%
spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]

#spk_count = 0
class fano_mean_match:
    
    def __init__(self):
        self.spk_sparmat = None
        self.stim_onoff = None#data.a1.param.stim.stim_on[20:40].copy() # ms stimulus onset, off       
        self.t_bf = 100 # ms; time before stimulus onset
        self.t_aft = 100 # ms; time after stimulus off
        self.dt = 0.1 # ms
        self.win = 150 # ms sliding window
        self.move_step = 10 # ms sliding window moving step
        self.repeat = 100
        
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
        spk_count = np.zeros([spk_sparmat.shape[0],samp_stp.shape[0],samp_onoff.shape[0]])
        for i in range(samp_onoff.shape[0]):
            samp_stp = np.arange(0, stim_onoff[0,1] - stim_onoff[0,0] + t_bf + t_aft + 1, move_step)
            samp_stp += samp_onoff[i,0]
            for stp in range(len(samp_stp)):
                spk_count[:,stp,i] = spk_sparmat[:, samp_stp[stp]-hfwin:samp_stp[stp]+hfwin].A.sum(1)
        
        mean_var = np.zeros([spk_sparmat.shape[0],samp_stp.shape[0],2])
        
        mean_var[:,:,0] = spk_count.mean(2)
        mean_var[:,:,1] = spk_count.var(2)
        
        bin_count = np.arange(0.0001, spk_count.max()+1, 0.3) 
        mean_hist = np.zeros([bin_count.shape[0]-1, samp_stp.shape[0]], int)
        for t in range(samp_stp.shape[0]):
            hist, _ = np.histogram(mean_var[:,t,0],bin_count)
            mean_hist[:,t] = hist
        
        common_mean_hist = mean_hist.min(1)
        #samp_ratio = common_mean_hist.reshape(-1,1)/mean_hist
        
        #samp_ratio[np.isnan(samp_ratio)] = 0
        bin_count_interval = bin_count[1] -bin_count[0]
        bin_non0 = bin_count[np.where(common_mean_hist != 0)[0]]
        common_mean_hist[common_mean_hist != 0]
        
        #repeat = 100
        fano_record = np.zeros([self.repeat, samp_stp.shape[0]])
        for t in range(samp_stp.shape[0]):
            #bin_non0 = bin_count[np.where(samp_ratio[:,t] != 0)[0]]
            #samp_ratio_non0 = samp_ratio[np.where(samp_ratio[:,t] != 0)[0]]
            for rpt in range(self.repeat):
                neu = [None]*len(bin_non0)
                for b in range(len(bin_non0)):
                    neu_chs = np.where((mean_var[:,t,0] >= bin_non0[b]) & (mean_var[:,t,0] < (bin_non0[b]+bin_count_interval)))[0]
                    neu_chs = np.random.choice(neu_chs, common_mean_hist[b],replace=False)
                    neu[b] = neu_chs
                neu = np.concatenate((neu))
                fano = (mean_var[neu,t,1]/mean_var[neu,t,0]).mean()
                fano_record[rpt, t] = fano
        
        fano_mean = fano_record.mean(0)
        fano_std = fano_record.std(0)
        
        return fano_mean, fano_std, fano_record
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
simu_time_tot = data.param.simutime#29000

#N_stim = data.a1.param.stim.stim_amp_scale.shape[0]

data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
#%%
stim_loc = np.array([0,0])

neuron = np.arange(data.a1.param.Ne)

dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,63,stim_loc)
neu_pool = [None]*1
neu_pool[0] = neuron[(dist >= 0) & (dist <= 10)]
#%%
data.a1.ge.spk_matrix[neu_pool[0],:]

#%%
#data.a1.param.stim.stim_on[i,0]

#spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]

for loop_num in range(1):
    data.load(datapath+'data%d.file'%loop_num)
    data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])

    fanomm = fano_mean_match.fano_mean_match()
    fanomm.bin_count_interval = 0.5
    fanomm.spk_sparmat = data.a1.ge.spk_matrix[neu_pool[0],:]
    fanomm.stim_onoff = data.a1.param.stim.stim_on[0:20].copy()
    fanomm.method = 'regression' # 'mean' or 'regression'
    fano_mean_noatt, fano_std_noatt, _ = fanomm.get_fano()
    print(np.sum(np.isnan(_)))
    print(np.sum(np.isnan(fano_mean_noatt)))
    print(np.sum(np.isnan(fano_std_noatt)))
    fanomm.stim_onoff = data.a1.param.stim.stim_on[20:40].copy()
    
    fano_mean_att, fano_std_att, _ = fanomm.get_fano()
    print(np.sum(np.isnan(_)))
    
    
    #fig, ax = plt.subplots(1,1)
    ax.errorbar(np.arange(fano_mean_noatt.shape[0])*10-100,fano_mean_noatt,fano_std_noatt)
    ax.errorbar(np.arange(fano_mean_att.shape[0])*10-100,fano_mean_att,fano_std_att)









