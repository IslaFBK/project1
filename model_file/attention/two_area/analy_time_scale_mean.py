#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:42:03 2021

@author: shni2598
"""
from scipy.signal import find_peaks
#%%
data_dir = 'raw_data/'
analy_type = 'wkfb'
#datapath = data_dir
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/twoarea_hz/time_scale/' + data_dir

#data = mydata.mydata()
#data.load(datapath+'data%d.file'%loop_num)
data_anly = mydata.mydata()

#%%
for loop_num in range(5):
    data_anly.load(datapath+'data_anly%d.file'%loop_num)
    if loop_num == 0:
        coef_tau1_all = np.zeros([*data_anly.time_scale.coef_tau1.shape,5])
        coef_tau2_all = np.zeros([*data_anly.time_scale.coef_tau2.shape,5])
        coef_tau1_all[:] = np.nan
        coef_tau2_all[:] = np.nan
    
    coef_tau1_all[:,:,loop_num] = data_anly.time_scale.coef_tau1
    coef_tau2_all[:,:,loop_num] = data_anly.time_scale.coef_tau2
    
#%%
coef_tau1_mean = np.nanmean(coef_tau1_all[0,:,:],-1)
coef_tau1_sem = sem(coef_tau1_all[0,:,:],-1,nan_policy='omit')
coef_tau2_mean = np.nanmean(coef_tau2_all[0,:,:],-1)
coef_tau2_sem = sem(coef_tau2_all[0,:,:],-1,nan_policy='omit')

#%%
def damp_oscil(x, tau1, f1):
    return 1 * np.exp(-x/tau1) * np.cos(2*np.pi*f1*x + 0)        

def damp_oscil2(x, y0, tau1, f1):
    return y0 * np.exp(-x/tau1) * np.cos(2*np.pi*f1*x + 0)        

def exp(x, tau1):
    return 1 * np.exp(-x/tau1)   

# def damp_oscil(x, y0_1, tau1, f1, y0_2, tau2, f2):
#     return y0_1 * np.exp(-x/tau1) * np.cos(2*np.pi*f1*x + 0) + \
#         y0_2 * np.exp(-x/tau2) * np.cos(2*np.pi*f2*x + 0)

# def damp_oscil(x, y0_1, tau1, f1, y0_2, tau2):
#     return (y0_1 * np.exp(-x/tau1) + y0_2 * np.exp(-x/tau2)) * np.cos(2*np.pi*f1*x + 0)
        
        
t_delay_1 = np.arange(coef_tau1_mean.shape[0])*sample_interval
t_delay_2 = np.arange(coef_tau2_mean.shape[0])*sample_interval

#f = 2; tau = 5; phi = 0
#x = np.linspace(0,20,200)
# popt1, pcov1 = curve_fit(damp_oscil, t_delay_1/1000, coef_tau1_mean, p0=[0.3, 3])
# popt2, pcov2 = curve_fit(damp_oscil, t_delay_2/1000, coef_tau2_mean, p0=[0.3, 3])
popt1, pcov1 = curve_fit(damp_oscil2, t_delay_1/1000, coef_tau1_mean, p0=[1, 0.3, 3])
popt2, pcov2 = curve_fit(damp_oscil2, t_delay_2/1000, coef_tau2_mean, p0=[1, 0.3, 3])

#%%
fig, ax = plt.subplots(1,1)
loop_num = 1
sample_interval = 5
t_delay_1 = np.arange(coef_tau1_mean.shape[0])*sample_interval
t_delay_2 = np.arange(coef_tau2_mean.shape[0])*sample_interval

ax.errorbar(t_delay_1, coef_tau1_all[0,:,loop_num], coef_tau1_all[1,:,loop_num], \
                     fmt='--', c=clr[0], marker='.', label='sens')
ax.errorbar(t_delay_2, coef_tau2_all[0,:,loop_num], coef_tau2_all[1,:,loop_num], \
                     fmt='--', c=clr[1], marker='.', label='asso')
#%%
fig, ax = plt.subplots(1,1)

sample_interval = 5
t_delay_1 = np.arange(coef_tau1_mean.shape[0])*sample_interval
t_delay_2 = np.arange(coef_tau2_mean.shape[0])*sample_interval

ax.errorbar(t_delay_1, coef_tau2_mean, coef_tau2_sem, \
                     fmt='--', c=clr[0], marker='.', label='sens')
ax.errorbar(t_delay_2, coef_tau2_all[0,:,2], coef_tau2_all[1,:,2], \
                     fmt='--', c=clr[1], marker='.', label='asso')
#%%
def fit_envelope(t_delay, coef_tau_mean):
    peaks, _ = find_peaks(coef_tau_mean)
    peaks_abs, _ = find_peaks(np.abs(coef_tau_mean))
    
    t_peaks = np.concatenate(([0], t_delay[peaks]))
    t_peaks_abs = np.concatenate(([0], t_delay[peaks_abs]))
    v_peaks =  np.concatenate(([1], coef_tau_mean[peaks]))
    v_peaks_abs =  np.concatenate(([1], np.abs(coef_tau_mean)[peaks_abs]))
    
    popt, pcov = curve_fit(exp, t_peaks[:3]/1000, v_peaks[:3], p0=[0.3])
    popt_abs, pcov_abs = curve_fit(exp, t_peaks_abs[:5]/1000, v_peaks_abs[:5], p0=[0.3])
    
    return peaks, peaks_abs, t_peaks, t_peaks_abs, v_peaks, v_peaks_abs, popt, popt_abs
#%%
peaks_2, peaks_abs_2, t_peaks_2, t_peaks_abs_2, v_peaks_2, v_peaks_abs_2, popt_2, popt_abs_2 = \
    fit_envelope(t_delay_2, coef_tau2_mean)
    
peaks_1, peaks_abs_1, t_peaks_1, t_peaks_abs_1, v_peaks_1, v_peaks_abs_1, popt_1, popt_abs_1 = \
    fit_envelope(t_delay_1, coef_tau1_mean)

#%%

peaks, _ = find_peaks(coef_tau2_mean)
peaks_abs, _ = find_peaks(np.abs(coef_tau2_mean))
#%%
t_peaks = np.concatenate(([0], t_delay_2[peaks]))
t_peaks_abs = np.concatenate(([0], t_delay_2[peaks_abs]))
v_peaks =  np.concatenate(([1], coef_tau2_mean[peaks]))
v_peaks_abs =  np.concatenate(([1], np.abs(coef_tau2_mean)[peaks_abs]))
#%%
popt2, pcov2 = curve_fit(exp, t_peaks[:3]/1000, v_peaks[:3], p0=[0.3])
popt2_abs, pcov2_abs = curve_fit(exp, t_peaks_abs[:5]/1000, v_peaks_abs[:5], p0=[0.3])
#%%
peaks, _ = find_peaks(coef_tau1_mean)
peaks_abs, _ = find_peaks(np.abs(coef_tau1_mean))
#%%
t_peaks = np.concatenate(([0], t_delay_2[peaks]))
t_peaks_abs = np.concatenate(([0], t_delay_2[peaks_abs]))
v_peaks =  np.concatenate(([1], coef_tau1_mean[peaks]))
v_peaks_abs =  np.concatenate(([1], np.abs(coef_tau1_mean)[peaks_abs]))
#%%
popt2, pcov2 = curve_fit(exp, t_peaks[:3]/1000, v_peaks[:3], p0=[0.3])
popt2_abs, pcov2_abs = curve_fit(exp, t_peaks_abs[:5]/1000, v_peaks_abs[:5], p0=[0.3])


#%%
fig, ax = plt.subplots(2,1)

sample_interval = 5
t_delay_1 = np.arange(coef_tau1_mean.shape[0])*sample_interval
t_delay_2 = np.arange(coef_tau2_mean.shape[0])*sample_interval



ax[0].errorbar(t_delay_1, coef_tau1_mean, coef_tau1_sem, \
                     fmt='--', c=clr[0], marker='.', label='sens')
    
#fig, ax = plt.subplots(1,1)
ax[0].plot(t_peaks_1, v_peaks_1, 'x',c=clr[2],markersize=10, label='sens; peak')
ax[0].plot(t_peaks_abs_1, v_peaks_abs_1,'*', c=clr[3], markersize=10, label='sens; peak; abs')

v_peak_fit_1 = exp(t_delay_1/1000, *popt_1)
v_peak_abs_fit_1 = exp(t_delay_1/1000, *popt_abs_1)
ax[0].plot(t_delay_1, v_peak_fit_1, ':', label='sens; fit; tau:%.2f ms'%(popt_1[0]*1000))
ax[0].plot(t_delay_1, v_peak_abs_fit_1,':', label='sens; fit; abs; tau:%.2f ms'%(popt_abs_1[0]*1000))


ax[1].errorbar(t_delay_2, coef_tau2_mean, coef_tau2_sem, \
                     fmt='--', c=clr[1], marker='.', label='sens')
    
#fig, ax = plt.subplots(1,1)
ax[1].plot(t_peaks_2, v_peaks_2, 'x',c=clr[2],markersize=10, label='asso; peak')
ax[1].plot(t_peaks_abs_2, v_peaks_abs_2,'*',c=clr[3], markersize=10, label='asso; peak; abs')

v_peak_fit_2 = exp(t_delay_2/1000, *popt_2)
v_peak_abs_fit_2 = exp(t_delay_2/1000, *popt_abs_2)

ax[1].plot(t_delay_2, v_peak_fit_2, ':', label='asso; fit; tau:%.2f ms'%(popt_2[0]*1000))
ax[1].plot(t_delay_2, v_peak_abs_fit_2,':', label='asso; fit; abs; tau:%.2f ms'%(popt_abs_2[0]*1000))
ax[0].legend()
ax[1].legend()

fig.suptitle('time-scale of auto correlation of spontaneous activity\n in sensory and association area')
#%%
fig, ax = plt.subplots(1,1)

sample_interval = 5
t_delay_1 = np.arange(coef_tau1_mean.shape[0])*sample_interval
t_delay_2 = np.arange(coef_tau2_mean.shape[0])*sample_interval

ax.errorbar(t_delay_1, coef_tau2_mean, coef_tau2_sem, \
                     fmt='--', c=clr[0], marker='.', label='sens')
ax.plot(t_delay_1[peaks], coef_tau2_mean[peaks], 'x', c=clr[1])
#%%
fig, ax = plt.subplots(1,1)

sample_interval = 5
t_delay_1 = np.arange(coef_tau1_mean.shape[0])*sample_interval
t_delay_2 = np.arange(coef_tau2_mean.shape[0])*sample_interval

ax.errorbar(t_delay_1, coef_tau1_mean, coef_tau1_sem, \
                     fmt='--', c=clr[0], marker='.', label='sens')
ax.errorbar(t_delay_2, coef_tau2_mean, coef_tau2_sem, \
                     fmt='--', c=clr[1], marker='.', label='asso')

# ax.plot(t_delay_1, damp_oscil(t_delay_1/1000, *popt1), \
#         c=clr[0], label='exp(-x/tau)*np.cos(2*pi*f*x);tau:%.2f,f:%.2f'%(popt1[0]*1000, popt1[1]))  
# ax.plot(t_delay_2, damp_oscil(t_delay_2/1000, *popt2), \
#         c=clr[1], label='exp(-x/tau)*np.cos(2*pi*f*x);tau:%.2f,f:%.2f'%(popt2[0]*1000, popt2[1]))    
ax.plot(t_delay_1, damp_oscil2(t_delay_1/1000, *popt1), \
        c=clr[0], label='y0*exp(-x/tau)*np.cos(2*pi*f*x);y0:%.2f,tau:%.2f,f:%.2f'%(popt1[0],popt1[1]*1000, popt1[2]))  
ax.plot(t_delay_2, damp_oscil2(t_delay_2/1000, *popt2), \
        c=clr[1], label='y0*exp(-x/tau)*np.cos(2*pi*f*x);y0:%.2f,tau:%.2f,f:%.2f'%(popt2[0],popt2[1]*1000, popt2[2]))    

# ax.plot(t_delay_1, np.abs(hb1), \
#         c=clr[0], label='sens-fit')  
# ax.plot(t_delay_2, np.abs(hb2), \
#         c=clr[1], label='asso-fit')  
ax.legend()  

#plt.title('auto correlation of spontaneous\ninitial correlation = 1') 
plt.title('auto correlation of spontaneous\ninitial correlation not forced')  
 
#%%
    
data_anly.time_scale = mydata.mydata()
data_anly.time_scale.coef_tau1 = coef_tau1
data_anly.time_scale.coef_tau2 = coef_tau2