#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:22:39 2021

@author: shni2598
"""

#%%
spk_count = mat_test.copy()
spk_count = spk_count.astype(float,)
spk_count -= mat_test.mean(1).reshape(-1,1)
spk_count /= np.sqrt(np.sum(spk_count**2, 1)).reshape(-1,1)
#%%
#spk_count1=spk_count
spk_count1=spk_count+np.random.randint(0,5,(3,5))

spk_count2=spk_count+np.random.randint(0,5,(3,5))
#%%
spk_count1 = spk_count1.astype(float)
spk_count2 = spk_count2.astype(float)

#if self.pair == 'all':
spk_count1 -= spk_count1.mean(1).reshape(-1,1)
spk_count2 -= spk_count2.mean(1).reshape(-1,1)
spk_count1 /= np.sqrt(np.sum(spk_count1**2, 1)).reshape(-1,1)
spk_count2 /= np.sqrt(np.sum(spk_count2**2, 1)).reshape(-1,1)

corr = np.dot(spk_count1, spk_count2.T)

#%%
gau_rn = np.random.multivariate_normal([1, 2], cov=[[4,0.5*4],[0.5*4,4]], size=1000)
#%%
cov = np.zeros([4,4])
cov[np.triu_indices(4,1)] = 0.5*4
cov[np.tril_indices(4,-1)] = 0.5*4
cov[np.diag_indices(4)] = 4

gau_rn2 = np.random.multivariate_normal([1,2,1,2], cov=cov, size=1000)
#%%
spk_count1 = gau_rn2[:,:2].T#spk_count1.astype(float)
spk_count2 = gau_rn2[:,2:].T#spk_count2.astype(float)

#if self.pair == 'all':
spk_count1 -= spk_count1.mean(1).reshape(-1,1)
spk_count2 -= spk_count2.mean(1).reshape(-1,1)
spk_count1 /= np.sqrt(np.sum(spk_count1**2, 1)).reshape(-1,1)
spk_count2 /= np.sqrt(np.sum(spk_count2**2, 1)).reshape(-1,1)

corr = np.dot(spk_count1, spk_count2.T)
#%%
spk_count1 = gau_rn2.T#spk_count1.astype(float)
#spk_count2 = gau_rn[:,1]#spk_count2.astype(float)

#if self.pair == 'all':
spk_count1 -= spk_count1.mean(1).reshape(-1,1)
#spk_count2 -= spk_count2.mean(1).reshape(-1,1)
spk_count1 /= np.sqrt(np.sum(spk_count1**2, 1)).reshape(-1,1)
#spk_count2 /= np.sqrt(np.sum(spk_count2**2, 1)).reshape(-1,1)

corr = np.dot(spk_count1, spk_count1.T)
#%%
n_StimAmp = 4
n_perStimAmp = 50
stim_amp = 200*2**np.arange(n_StimAmp)

datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/ff/raw_data/'
#sys_argv = int(sys.argv[1])
loop_num = 16 #rep_ind*20 + ie_num
good_dir = 'good/'
goodsize_dir = 'good_size/'

savefile_name = 'data_anly' #'data_anly'

fftplot = 1; getfano = 1
get_nscorr = 1
get_TunningCurve = 1; get_HzTemp = 1

if loop_num%10 ==0: get_ani = 1
else: get_ani = 0

#%%
clr = plt.rcParams['axes.prop_cycle'].by_key()['color']

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

#%%
nscorr = fra.noise_corr()

neuron = np.arange(data.a1.param.Ne)
neu_pool = [None]*2
stim_loc = np.array([0,0])
neu_range = 5
dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]
stim_loc = np.array([-32,-32])
neu_range = 5
dist = cn.coordination.lattice_dist(data.a1.param.e_lattice,data.a1.param.width,stim_loc)
neu_pool[1] = neuron[(dist >= 0) & (dist <= neu_range)]
    # neu_pool = [None]*1
    # neu_range = 5
    # neu_pool[0] = neuron[(dist >= 0) & (dist <= neu_range)]

data_anly.nscorr = mydata.mydata()
data_anly.nscorr.neu_pool = neu_pool

simu_time_tot = data.param.simutime#29000
    
    #N_stim = data.a1.param.stim.stim_amp_scale.shape[0]
data.a1.ge.get_sparse_spk_matrix([data.a1.param.Ne, simu_time_tot*10])
nscorr.spk_matrix1 = data.a1.ge.spk_matrix[neu_pool[0],:]
nscorr.spk_matrix2 = data.a1.ge.spk_matrix[neu_pool[1],:]


ns = np.zeros([2,n_StimAmp,2,2]); #[pair,n_StimAmp,att,mean-sem]

#fig, ax = plt.subplots(1,1, figsize=[6,4])

for st in range(n_StimAmp):  
    '''no-att; within 1 group'''
    nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
    #nscorr.dura2 = None
    corr_mean, corr_sem = nscorr.get_nc_withingroup()
    ns[0, st, 0, 0], ns[0, st, 0, 1] = corr_mean, corr_sem
    '''att; within 1 group'''
    nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
    #nscorr.dura2 = None
    corr_mean, corr_sem = nscorr.get_nc_withingroup()
    ns[0, st, 1, 0], ns[0, st, 1, 1] = corr_mean, corr_sem
    '''no-att; between 2 groups'''
    nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
    #nscorr.dura2 = nscorr.dura1
    corr_mean, corr_sem = nscorr.get_nc_betweengroups()
    ns[1, st, 0, 0], ns[1, st, 0, 1] = corr_mean, corr_sem
    '''att; between 2 groups'''
    nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
    #nscorr.dura2 = nscorr.dura1
    corr_mean, corr_sem = nscorr.get_nc_betweengroups()
    ns[1, st, 1, 0], ns[1, st, 1, 1] = corr_mean, corr_sem

data_anly.nscorr.nscorr = ns

fig, ax = plt.subplots(1,1, figsize=[6,4])
ax.errorbar(np.arange(len(stim_amp)), ns[0,:,0,0], ns[0,:,0,1], c=clr[0], fmt='--', marker='o', label='no-att;1-group')
ax.errorbar(np.arange(len(stim_amp)), ns[0,:,1,0], ns[0,:,1,1], c=clr[1], fmt='-', marker='o', label='att;1-group')
ax.errorbar(np.arange(len(stim_amp)), ns[1,:,0,0], ns[1,:,0,1], c=clr[0], fmt='--', marker='x', label='no-att;2-group')
ax.errorbar(np.arange(len(stim_amp)), ns[1,:,1,0], ns[1,:,1,1], c=clr[1], fmt='-', marker='x', label='att;2-group')

#ax.legend()
ax.legend()
ax.xaxis.set_ticks([i for i in range(len(stim_amp))])
ax.xaxis.set_ticklabels([str(item) for item in stim_amp])     
#title3 = title + '_win%.1f_bin%d\n_range%d'%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
fig.suptitle(title)
savetitle = title.replace('\n','')
# nsfile = savetitle+'_nc_%d'%(loop_num)+'.png'
# fig.savefig(nsfile)
# plt.close()
#%%
nscorr.win = 100 # ms sliding window length to count spikes
nscorr.move_step = 20 # ms sliding window move step, (sampling interval for time varying noise correlation)
nscorr.t_bf = -nscorr.win/2 # ms; time before stimulus onset to start to sample noise correlation
nscorr.t_aft = -nscorr.win/2 # ms; time after stimulus off to finish sampling noise correlation
#%%
st = 3
nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
corr_ = nscorr.get_nc_withingroup_t()
#%%
for st in range(4):
    nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
    corr_ = nscorr.get_nc_withingroup_t()
    print(corr_)
#%%
ns_t = np.zeros([2,2,corr_.shape[1],n_StimAmp*2])
for st in range(n_StimAmp):  
    '''no-att; within 1 group'''
    nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
    #nscorr.dura2 = None
    corr_ = nscorr.get_nc_withingroup_t()
    if st == 0:
        ns_t = np.zeros([2,2,corr_.shape[1],n_StimAmp*2])

    ns_t[0, :, :, st] = corr_
    '''att; within 1 group'''
    nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
    #nscorr.dura2 = None
    corr_ = nscorr.get_nc_withingroup_t()
    ns_t[0, :, :, st+n_StimAmp] = corr_
    '''no-att; between 2 groups'''
    nscorr.dura1 = data.a1.param.stim1.stim_on[st*n_perStimAmp:(st+1)*n_perStimAmp]
    #nscorr.dura2 = nscorr.dura1
    corr_ = nscorr.get_nc_betweengroups_t()
    ns_t[1, :, :, st] = corr_
    '''att; between 2 groups'''
    nscorr.dura1 = data.a1.param.stim1.stim_on[(st+n_StimAmp)*n_perStimAmp:(st+n_StimAmp+1)*n_perStimAmp]
    #nscorr.dura2 = nscorr.dura1
    corr_ = nscorr.get_nc_betweengroups_t()
    ns_t[1, :, :, st+n_StimAmp] = corr_
#%%
data_anly.nscorr.nscorr = ns
fig, ax = plt.subplots(2,1, figsize=[8,6])
sample_t = np.arange(ns_t.shape[2])*nscorr.move_step-nscorr.t_bf
for st in range(n_StimAmp):  
    ax[0].errorbar(sample_t, ns_t[0, 0, :, st],ns_t[0, 1, :, st], c=clr[st], fmt='--', marker='o', label='no-att;1-group;amp:%.1fHz'%stim_amp[st])
    ax[0].errorbar(sample_t, ns_t[0, 0, :, st+n_StimAmp],ns_t[0, 1, :, st+n_StimAmp], c=clr[st], fmt='-', marker='o', label='att;1-group;amp:%.1fHz'%stim_amp[st])
    ax[1].errorbar(sample_t, ns_t[1, 0, :, st],ns_t[1, 1, :, st], c=clr[st], fmt='--', marker='x', label='no-att;2-group;amp:%.1fHz'%stim_amp[st])
    ax[1].errorbar(sample_t, ns_t[1, 0, :, st+n_StimAmp],ns_t[1, 1, :, st+n_StimAmp], c=clr[st], fmt='-', marker='x', label='att;2-group;amp:%.1fHz'%stim_amp[st])

#ax.legend()
ax[0].legend()
ax[0].set_xlim([sample_t.min()-20,sample_t.max()+150])
ax[1].legend()
ax[1].set_xlim([sample_t.min()-20,sample_t.max()+150])

#%%
# ax.xaxis.set_ticks([i for i in range(len(stim_amp))])
# ax.xaxis.set_ticklabels([str(item) for item in stim_amp])     
#title3 = title + '_win%.1f_bin%d\n_range%d'%(fanomm.win, bin_count_interval_hz, neu_range)#%(data_anly.fano_noatt.mean(), data_anly.fano_att.mean())
fig.suptitle(title)
savetitle = title.replace('\n','')
# nsfile = savetitle+'_nc_%d'%(loop_num)+'.png'
# fig.savefig(nsfile)
# plt.close()








