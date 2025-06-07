#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:32:39 2021

@author: shni2598
"""

hz_t_bi = np.zeros(hz_t.shape)
hz_t_bi[hz_t>20]=1
hz_t_bi_off = np.zeros(hz_t_bi.shape)
hz_t_bi_off[:,1:] = hz_t_bi[:,0:-1]

on_off = hz_t_bi - hz_t_bi_off
#%%

time1 = np.where(on_off[0])
time1_on = time1[0][0::2]
time1_off = time1[0][1::2]

dura1 = (time1_off - time1_on)

time2 = np.where(on_off[1])
time2_on = time2[0][0::2]
time2_off = time2[0][1::2]

dura2 = (time2_off - time2_on)
#%%
fig,ax = plt.subplots(1,1)
#plt.plot(dura1)
ax.hist(dura1)
ax.set_yscale('log')

ax.hist(dura2)
ax.set_yscale('log')
#%%
loop_num = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/fb/fbnum/raw_data/' 

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()
#%%
dt = 1/10000;
end = int(20/dt); start = int(5/dt)
spon_rate_e1 = np.sum((data.a1.ge.t < end) & (data.a1.ge.t >= start))/15/data.a1.param.Ne
spon_rate_i1 = np.sum((data.a1.gi.t < end) & (data.a1.gi.t >= start))/15/data.a1.param.Ni
#%%
spon_rate_e2 = np.sum((data.a2.ge.t < end) & (data.a2.ge.t >= start))/15/data.a2.param.Ne
spon_rate_i2 = np.sum((data.a2.gi.t < end) & (data.a2.gi.t >= start))/15/data.a2.param.Ni
#%%







sig = 6
tau = 10
tau2 = 15
x = np.linspace(-20,20,60)
coupling = np.exp(-x/tau)
coupling2 = np.exp(-1/2*(x/tau2)**2)

coupling[:30] = coupling[30:][::-1]
coupling2[:30] = coupling2[30:][::-1]

#%%
r = np.exp(-1/2*(x/sig)**2)
o = np.convolve(r,coupling, 'same')
o2 = np.convolve(r,coupling2, 'same')

#%%
plt.figure()
plt.plot(r/r.max())
plt.plot(o2/o2.max())
plt.plot(o/o.max(),'--')

#plt.plot(coupling2)

#%%
loop_num = 0
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/fb/movie1/raw_data/' 

data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()
#%%
start_time = 5e3; end_time = 20e3
window = 5
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=50, n_neuron = data.a1.param.Ne, window = 50)
data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=50, n_neuron = data.a2.param.Ne, window = 50)

#%%
sua1 = data.a1.ge.spk_rate.spk_rate.reshape(data.a1.param.Ne,-1).astype(int)
sua2 = data.a2.ge.spk_rate.spk_rate.reshape(data.a2.param.Ne,-1).astype(int)

#%%
xcorr1 = np.zeros(2*sua1.shape[1]-1)
xcorr2 = np.zeros(2*sua2.shape[1]-1)
#%%
for i in range(data.a1.param.Ne):
    xcorr1 += np.correlate(sua1[i],sua1[i],'full')/np.sqrt(np.sum(sua1[i]**2))
    xcorr2 += np.correlate(sua2[i],sua2[i],'full')/np.sqrt(np.sum(sua2[i]**2))
#%%
xcorr1 /= data.a1.param.Ne
xcorr2 /= data.a2.param.Ne

#%%
plt.figure()
plt.plot(xcorr1)
plt.plot(xcorr2)
#%%
sua1_trial = np.reshape(sua1, (data.a1.param.Ne, 15, -1))
sua2_trial = np.reshape(sua2, (data.a2.param.Ne, 15, -1))
#%%
coef1 = np.zeros([data.a1.param.Ne, sua1_trial.shape[-1], sua1_trial.shape[-1]])
coef1 = np.zeros([data.a1.param.Ne, sua1_trial.shape[-1], sua1_trial.shape[-1]])

for i in range(data.a1.param.Ne):
    coef1[i] = np.corrcoef(sua1_trial[i].T)
#%%
coef_tau = np.zeros([6])
for i in range(1,7):
    coef_tau[i-1] = np.nanmean(coef1.diagonal(i,1,2))
#%%
def time_scale(sua_trial, delay_max=6):
    coef = np.zeros([sua_trial.shape[0], sua_trial.shape[-1], sua_trial.shape[-1]])
    for i in range(sua_trial.shape[0]):
        coef[i] = np.corrcoef(sua_trial[i].T)
    coef_tau = np.zeros([delay_max])
    for i in range(1,delay_max+1):
        coef_tau[i-1] = np.nanmean(coef.diagonal(i,1,2))
    return coef_tau, coef
#%%
coef_tau1, coef1 = time_scale(sua1_trial)
coef_tau2, coef2 = time_scale(sua2_trial)

    #coef = np.zeros([sua1_trial.shape[0], sua2_trial.shape[-1], sua2_trial.shape[-1]])
#%%
    
    
    
plt.figure()
plt.plot(coef_tau1)
plt.plot(coef_tau2)








