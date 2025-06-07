#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:38:06 2020

@author: shni2598
"""
import matplotlib as mpl
mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
import post_analysis as psa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import levy
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/gc_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_gc_AD = np.zeros([10,20])

#%%
for loop_num in range(200): # sys.argv[1]
    #ee_ind = loop_num//20%10
    ie_ind  = loop_num%20
    rep_ind = loop_num//20%10
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_gc_AD[rep_ind, ie_ind] = spkdata.a1.ge.rate_overall 

#%%
#ref = np.linspace(5,5.5,5)
ie = np.linspace(2.4, 4.5, 20)
#%%
#ref_num = 4
rep_ind = 0
ie_num = 3
#otr_num = 0
print(ref[ref_num], ie[ie_num])
#%%
plt.figure()
#plt.errorbar(ie, rate_mat_gc_AD.mean(0), yerr=rate_mat_gc_AD.std(0), fmt='*')
#plt.plot(ie, rate_mat_pw_rep_2[3,:], '*')
plt.plot(ie, rate_mat_gc_AD.mean(0), '*')
#plt.errorbar(ie, rate_mat_pw_rep.mean(0), yerr=rate_mat_pw_rep.std(0), fmt='.')
plt.yscale('log')

#%%
from scipy.optimize import curve_fit
#%%
def power_law(x, a, b, c):
    return a*x**b + c
#%%
pars_1_gc_AD, cov_1_gc_AD = curve_fit(power_law, ie[2:11],rate_mat_gc_AD[:,2:11].mean(0))   
pars_2_gc_AD, cov_2_gc_AD = curve_fit(power_law, ie[11:],rate_mat_gc_AD[:,11:].mean(0))   

#%%
plt.figure()

y_1 = power_law(ie[2:11], *pars_1_gc_AD)
y_2 = power_law(ie[11:], *pars_2_gc_AD)
#plt.figure()
plt.plot(ie[2:11], y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1_gc_AD[0],pars_1_gc_AD[1],pars_1_gc_AD[2]))
plt.plot(ie[11:], y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_2_gc_AD[0],pars_2_gc_AD[1],pars_2_gc_AD[2]))
plt.errorbar(ie, rate_mat_gc_AD.mean(0), yerr=rate_mat_gc_AD.std(0), fmt='*')
plt.yscale('log')
plt.legend()
#plt.title("Adam synapse model, yifan's params\n e-i:%.5fuS, i-i:%.5fuS\ncan produce plane wave"%(spkdata.a1.param.w_ei,spkdata.a1.param.w_ii))
plt.title("Adam synapse model, guozhang's params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms\ncan't produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di))

plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_AD_gc_param_1.svg')
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/gc_param/'

loop_num = 81 #rep_ind*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.ge.t = spkdata.a1.ge.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 10*second
binforrate=5*ms
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkdata.a1.ge, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkdata.a2.e, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#%%

spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)

#spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)
#%%
anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 1000; endstep = 2000
frames = endstep - startstep
start_time = startstep
ani1 = psa.show_pattern(spkrate1[:,:,startstep:endstep], spkrate2=1, area_num = 1, frames = frames, start_time = start_time, anititle=anititle)


#%%
'''
YG/gc_param
dgk_ind = 0 : delta_gk = 3nS
dgk_ind = 1 : delta_gk = 10nS

'''

datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/gc_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_gc_YG = np.zeros([2,20])

#%%
for loop_num in range(40): # sys.argv[1]
    #ee_ind = loop_num//20%10
    ie_ind  = loop_num%20
    dgk_ind = loop_num//20%2
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_gc_YG[dgk_ind, ie_ind] = spkdata.a1.ge.rate_overall 

#%%
#ref = np.linspace(5,5.5,5)
ie = np.linspace(2.4, 4.5, 20)
#%%
#ref_num = 4
rep_ind = 0
ie_num = 3
#otr_num = 0
print(ref[ref_num], ie[ie_num])
#%%
plt.figure()
#plt.errorbar(ie, rate_mat_gc_AD.mean(0), yerr=rate_mat_gc_AD.std(0), fmt='*')
#plt.plot(ie, rate_mat_pw_rep_2[3,:], '*')
plt.plot(ie, rate_mat_gc_YG[1,:], '*')
#plt.errorbar(ie, rate_mat_pw_rep.mean(0), yerr=rate_mat_pw_rep.std(0), fmt='.')
plt.yscale('log')

#%%
from scipy.optimize import curve_fit
#%%
def power_law(x, a, b, c):
    return a*x**b + c
#%%
pars_1_gc_YG, cov_1_gc_YG = curve_fit(power_law, ie[2:10],rate_mat_gc_YG[0,2:10])   
pars_2_gc_YG, cov_2_gc_YG = curve_fit(power_law, ie[11:],rate_mat_gc_YG[0,11:])   

#%%
plt.figure()

y_1 = power_law(ie[2:11], *pars_1_gc_AD)
y_2 = power_law(ie[11:], *pars_2_gc_AD)
#plt.figure()
plt.plot(ie[2:11], y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1_gc_AD[0],pars_1_gc_AD[1],pars_1_gc_AD[2]))
plt.plot(ie[11:], y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_2_gc_AD[0],pars_2_gc_AD[1],pars_2_gc_AD[2]))
plt.errorbar(ie, rate_mat_gc_AD.mean(0), yerr=rate_mat_gc_AD.std(0), fmt='*')
plt.yscale('log')
plt.legend()
#plt.title("Adam synapse model, yifan's params\n e-i:%.5fuS, i-i:%.5fuS\ncan produce plane wave"%(spkdata.a1.param.w_ei,spkdata.a1.param.w_ii))
plt.title("Adam synapse model, guozhang's params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms\ncan produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di))

plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_AD_gc_param_1.svg')
#%%

datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/gc_param/'

loop_num = 1 #rep_ind*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.ge.t = spkdata.a1.ge.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 10*second
binforrate=5*ms
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkdata.a1.ge, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkdata.a2.e, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#%%

spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)

#spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)
#%%
anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 2000; endstep = 3000
frames = endstep - startstep
start_time = startstep
ani1 = psa.show_pattern(spkrate1[:,:,startstep:endstep], spkrate2=1, area_num = 1, frames = frames, start_time = start_time, anititle=anititle)
#%%
'''
guozhang's criticality param
'''

#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/gc_param/gc_param_crti/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_gc_AD_crti = np.zeros([10,18])

#%%
for loop_num in range(180): # sys.argv[1]
    #ee_ind = loop_num//20%10
    ie_ind  = loop_num%18
    rep_ind = loop_num//18%10
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_gc_AD_crti[rep_ind, ie_ind] = spkdata.a1.ge.rate_overall 

#%%
#ref = np.linspace(5,5.5,5)
ie = 3.375*(np.arange(0.7,1.56,0.05)-0.02)
#%%
#ref_num = 4
rep_ind = 0
ie_num = 3
#otr_num = 0
print(ref[ref_num], ie[ie_num])
#%%
plt.figure()
#plt.errorbar(ie, rate_mat_gc_AD.mean(0), yerr=rate_mat_gc_AD.std(0), fmt='*')
#plt.plot(ie, rate_mat_pw_rep_2[3,:], '*')
plt.plot(ie, rate_mat_gc_AD_crti.mean(0), '*')
#plt.errorbar(ie, rate_mat_pw_rep.mean(0), yerr=rate_mat_pw_rep.std(0), fmt='.')
plt.yscale('log')

#%%
from scipy.optimize import curve_fit
#%%
def power_law(x, a, b, c):
    return a*x**b + c
#%%
def exponential(x, a, b, c):
    return a*np.exp(b*x) + c
#%%
pars_1_gc_AD_crti, cov_1_gc_AD_crti = curve_fit(power_law, ie[2:9],rate_mat_gc_AD_crti[:,2:9].mean(0))   
pars_2_gc_AD_crti, cov_2_gc_AD_crti = curve_fit(power_law, ie[10:],rate_mat_gc_AD_crti[:,10:].mean(0),p0=np.array([13353,-7,1]))   
#%%
pars_1_gc_AD_crti, cov_1_gc_AD_crti = curve_fit(exponential, ie[4:10],rate_mat_gc_AD_crti[:,4:10].mean(0))   
pars_2_gc_AD_crti, cov_2_gc_AD_crti = curve_fit(exponential, ie[11:],rate_mat_gc_AD_crti[:,11:].mean(0))   


#%%
plt.figure()

y_1 = power_law(ie[2:9], *pars_1_gc_AD_crti)
y_2 = power_law(ie[10:], *pars_2_gc_AD_crti)
#plt.figure()
plt.plot(ie[2:9], y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1_gc_AD_crti[0],pars_1_gc_AD_crti[1],pars_1_gc_AD_crti[2]))
plt.plot(ie[10:], y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_2_gc_AD_crti[0],pars_2_gc_AD_crti[1],pars_2_gc_AD_crti[2]))
plt.errorbar(ie, rate_mat_gc_AD_crti.mean(0), yerr=rate_mat_gc_AD_crti.std(0), fmt='*')
plt.yscale('log')
plt.xscale('log')

plt.legend()
#plt.title("Adam synapse model, yifan's params\n e-i:%.5fuS, i-i:%.5fuS\ncan produce plane wave"%(spkdata.a1.param.w_ei,spkdata.a1.param.w_ii))
plt.title("Adam synapse model, guozhang's 'theta-gamma' params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms, tau_s_de:%.1fms\ncan't produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di,spkdata.a1.param.tau_s_de))

plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_AD_gc_param_theta-gamma.svg')
#%%
np.sum((rate_mat_gc_AD_crti[:,10:].mean(0)-y_2)**2)
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/gc_param/gc_param_crti/'

loop_num = 6 #rep_ind*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.ge.t = spkdata.a1.ge.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 10*second
binforrate=5*ms
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkdata.a1.ge, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkdata.a2.e, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#%%

spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)

#spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)
#%%
anititle="Adam synapse model, guozhang's 'theta-gamma' params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms, tau_s_de:%.1fms\ncan't produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di,spkdata.a1.param.tau_s_de)#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 2000; endstep = 4000
frames = endstep - startstep
start_time = startstep
ani1 = psa.show_pattern(spkrate1[:,:,startstep:endstep], spkrate2=1, area_num = 1, frames = frames, start_time = start_time, anititle=anititle)

ani1.save('AD_gc_param_theta-gamma.mp4')
#%%
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/gc_param/gc_param_crti/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_gc_YG_crti = np.zeros([10,18])

#%%
for loop_num in range(180): # sys.argv[1]
    #ee_ind = loop_num//20%10
    ie_ind  = loop_num%18
    rep_ind = loop_num//18%10
    try:
        with open(datapath+'data%d.file'%loop_num, 'rb') as file:
            #data = pickle.load(file)
            spkdata = mydata.mydata(pickle.load(file))
        rate_mat_gc_YG_crti[rep_ind, ie_ind] = spkdata.a1.ge.rate_overall 

    except: continue
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969

#%%
#ref = np.linspace(5,5.5,5)
ie = 3.375*(np.arange(0.7,1.56,0.05)-0.02)
#%%
#ref_num = 4
rep_ind = 0
ie_num = 3
#otr_num = 0
print(ref[ref_num], ie[ie_num])
#%%
plt.figure()
#plt.errorbar(ie, rate_mat_gc_AD.mean(0), yerr=rate_mat_gc_AD.std(0), fmt='*')
#plt.plot(ie, rate_mat_pw_rep_2[3,:], '*')
plt.plot(ie, rate_mat_gc_YG_crti.mean(0), '*')
#plt.plot(ie, rate_mat_gc_YG_crti[2,:], '*')
#plt.errorbar(ie, rate_mat_pw_rep.mean(0), yerr=rate_mat_pw_rep.std(0), fmt='.')
plt.yscale('log')

#%%
from scipy.optimize import curve_fit
#%%
def power_law(x, a, b, c):
    return a*x**b + c
#%%
def exponential(x, a, b, c):
    return a*np.exp(b*x) + c
#%%
pars_1_gc_YG_crti, cov_1_gc_YG_crti = curve_fit(power_law, ie[1:7],rate_mat_gc_YG_crti[:,1:7].mean(0))   
pars_2_gc_YG_crti, cov_2_gc_YG_crti = curve_fit(power_law, ie[9:],rate_mat_gc_YG_crti[:,9:].mean(0),p0=np.array([13353,-7,1]))   
#%%
pars_1_gc_AD_crti, cov_1_gc_AD_crti = curve_fit(exponential, ie[4:10],rate_mat_gc_AD_crti[:,4:10].mean(0))   
pars_2_gc_AD_crti, cov_2_gc_AD_crti = curve_fit(exponential, ie[11:],rate_mat_gc_AD_crti[:,11:].mean(0))   


#%%
plt.figure()

y_1 = power_law(ie[1:7], *pars_1_gc_YG_crti)
y_2 = power_law(ie[9:], *pars_2_gc_YG_crti)
#plt.figure()
plt.plot(ie[1:7], y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1_gc_YG_crti[0],pars_1_gc_YG_crti[1],pars_1_gc_YG_crti[2]))
plt.plot(ie[9:], y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_2_gc_YG_crti[0],pars_2_gc_YG_crti[1],pars_2_gc_YG_crti[2]))
plt.errorbar(ie, rate_mat_gc_YG_crti.mean(0), yerr=rate_mat_gc_YG_crti.std(0), fmt='*')
plt.yscale('log')
plt.xscale('log')

plt.legend()
#plt.title("Adam synapse model, yifan's params\n e-i:%.5fuS, i-i:%.5fuS\ncan produce plane wave"%(spkdata.a1.param.w_ei,spkdata.a1.param.w_ii))
plt.title("Yifan synapse model, guozhang's 'theta-gamma' params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms, tau_s_de:%.1fms\ncan produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di,spkdata.a1.param.tau_s_de))

plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_YG_gc_param_theta-gamma.svg')
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/gc_param/gc_param_crti/'

loop_num = 6 #rep_ind*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.ge.t = spkdata.a1.ge.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 10*second
binforrate=10*ms
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkdata.a1.ge, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkdata.a2.e, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#%%
tic = time.perf_counter()
spkrate1 = psa.get_spike_rate_sp(spkdata.a1.ge, starttime, endtime, indiv_rate = True, popu_rate = False, \
                   sample_interval = 1*ms, n_neuron = 3969, window = 10*ms, dt = 0.1*ms)
print(time.perf_counter() - tic)
#%%
tic = time.perf_counter()
spkrate2 = psa.get_spike_rate(spkdata.a1.ge, starttime, endtime, indiv_rate = True, popu_rate = False, \
                   sample_interval = 1*ms, n_neuron = 3969, window = 10*ms, dt = 0.1*ms)
print(time.perf_counter() - tic)
#%%

#%%
import firing_rate
#%%
spkdata.a1.ge.t = np.round(spkdata.a1.ge.t/(0.1*ms)).astype(int)
#%%
tic = time.perf_counter()
starttime2 = 0; endtime2 = 10e3
spkrate3,pop_r = firing_rate.get_spike_rate(spkdata.a1.ge, starttime2, endtime2, indiv_rate = True, popu_rate = True, \
                   sample_interval = 1, n_neuron = 3969, window = 10, dt = 0.1)
print(time.perf_counter() - tic)
#%%
spkrate3 = psa.overlap_centreandspike(centre_ind1, spkrate3, show_trajectory = False)

#spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)
#%%
anititle="Yifan synapse model, guozhang's 'theta-gamma' params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms, tau_s_de:%.1fms\ncan produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di,spkdata.a1.param.tau_s_de)#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 2000; endstep = 4000
frames = endstep - startstep
start_time = startstep
ani1 = psa.show_pattern(spkrate3[:,:,startstep:endstep], spkrate2=1, area_num = 1, frames = frames, start_time = start_time, anititle=anititle)

#ani1.save('YG_gc_param_theta-gamma.mp4')
#%%
tic = time.perf_counter()

spkdata.a1.ge.get_spike_rate(start_time=0, end_time=10e3,)
print(time.perf_counter() - tic)

spkdata.a1.ge.get_centre_mass()
spkdata.a1.ge.overlap_centreandspike()
#%%
anititle="Yifan synapse model, guozhang's 'theta-gamma' params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms, tau_s_de:%.1fms\ncan produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di,spkdata.a1.param.tau_s_de)#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 2000; endstep = 4000
frames = endstep - startstep
start_time = startstep
ani1 = firing_rate.show_pattern(spkdata.a1.ge.spk_rate.spk_rate[:,:,startstep:endstep], frames = frames, start_time = start_time, anititle=anititle)
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/gc_param/gc_param_crti/'

loop_num = 6 #rep_ind*20 + ie_num
#%%
spkdata2 = mydata.mydata()
spkdata2.load(datapath+'data%d.file'%loop_num)
#%%
spkdata2.a1.ge.get_spike_rate(start_time=0, end_time=10e3,)
spkdata2.a1.ge.get_centre_mass()
spkdata2.a1.ge.overlap_centreandspike()
#%%
anititle="Yifan synapse model, guozhang's 'theta-gamma' params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms, tau_s_de:%.1fms\ncan produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di,spkdata.a1.param.tau_s_de)#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 2000; endstep = 4000
frames = endstep - startstep
start_time = startstep
ani1 = firing_rate.show_pattern(spkdata2.a1.ge.spk_rate.spk_rate[:,:,startstep:endstep], frames = frames, start_time = start_time, anititle=anititle)
#%%
fig, ax = plt.subplots(1,1)
firing_rate.plot_traj(ax, spkdata2.a1.ge.centre_mass.centre_ind[:500])
ax.set_xlim([-0.5,62.5])
ax.set_ylim([-0.5,62.5])
#%%
datadict = spkdata2.class2dict()
#spkdata2.save(datadict, 'mysavefunc.file')
#%%
spkdata3 = mydata.mydata()
spkdata3.load('mysavefunc.file')
#%%
spkdata2.a1.ge.get_spike_rate(start_time=0, end_time=10e3,window=3)
spkdata2.a1.ge.get_pop_rate(0,10e3,sample_interval=1,window=5)
#%%
plt.figure()
plt.plot(spkdata2.a1.ge.pop_rate.pop_rate)
#%%
fftresult = np.fft.fft(spkdata2.a1.ge.pop_rate.pop_rate)
#%%
plt.figure()
plt.loglog(np.arange(int(len(fftresult)/2))/(len(fftresult))*1000,abs(fftresult[:int(len(fftresult)/2)]))
#%%
mua_loca = [0, 0]
mua_range = 6
mua_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.lattice_ext, mua_loca, mua_range, ijwd.width)
#%%
mua = spkdata2.a1.ge.spk_rate.spk_rate.reshape(3969,-1)[mua_neuron].mean(0)
#%%
plt.figure()
plt.plot(mua)

muafft = np.fft.fft(mua)
muafft[0] = 0
plt.figure()
plt.loglog(np.arange(len(muafft)//2)/len(muafft)*1000, abs(muafft[:len(muafft)//2]))
#%%
plt.figure()
plt.plot(np.arange(len(muafft)//2)/len(muafft)*1000, abs(muafft[:len(muafft)//2]))

#%%
with open('saveclass.file', 'wb') as file:
    #data = pickle.load(file)
    pickle.dump(spkdata2,file)

#%%
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

#%%
get_size(spkdata2)
#%%
import firing_rate_analysis
#%%
#spkdata2.a1.ge.get_spike_rate(start_time=0, end_time=10e3,window=3)

spk_rate_func = firing_rate_analysis.get_centre_mass(spkdata2.a1.ge.spk_rate.spk_rate, 1, 1, 15)
#%%
spkrate_func = firing_rate_analysis.overlap_centreandspike(spk_rate_func[0], spkdata2.a1.ge.spk_rate.spk_rate, show_trajectory = False)
#%%
anititle="Yifan synapse model, guozhang's 'theta-gamma' params\n p_ee:%.2f, dgk:%dnS, tau_s_di:%.1fms, tau_s_de:%.1fms\ncan produce plane wave"%(spkdata.a1.param.p_ee,spkdata.a1.param.delta_gk,spkdata.a1.param.tau_s_di,spkdata.a1.param.tau_s_de)#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 2000; endstep = 4000
frames = endstep - startstep
start_time = startstep
ani1 = firing_rate_analysis.show_pattern(spkrate_func[:,:,startstep:endstep], frames = frames, start_time = start_time, anititle=anititle)
#%%
spkdata2.a1.ge.get_spike_rate(start_time=0,end_time=10e3)

