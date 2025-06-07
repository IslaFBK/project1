#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:53:16 2020

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
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat = np.zeros([10,20,2])

#%%
for loop_num in range(400): # sys.argv[1]
    ee_ind = loop_num//20%10
    ie_ind  = loop_num%20 
    otr_ind = loop_num//200%2
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat[ee_ind, ie_ind, otr_ind] = spkdata.a1.e.rate_overall 

#%%
loop_num = 1
with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))
    

spkdata.a1.e.rate_overall

#%%
ee = np.linspace(0.7,1.1,10)
ie = np.linspace(2.4, 4.5, 20)
#%%
ee_num = 9
ie_num = 3
print(ee[ee_num], ie[ie_num])
#%%
plt.figure()
plt.plot(ie, np.log10(rate_mat[ee_num,:,0]), '*')
#%%
loop_num = 1*200 + ee_num*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.e.t = spkdata.a1.e.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 10*second
binforrate=5*ms
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkdata.a1.e, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkdata.a2.e, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#%%

spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)

#spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)
#%%
anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 0; endstep = 1000
frames = endstep - startstep
start_time = startstep
ani1 = psa.show_pattern(spkrate1[:,:,startstep:endstep], spkrate2=1, area_num = 1, frames = frames, start_time = start_time, anititle=anititle)
#%%
spkdata.a1.e.i.size/10/3969
spkdata.a1.param.__dict__
#%%
levy.fit_levy(jump_size1[:,0])
#%%
plt.figure()
plt.plot(spkdata.a1.e.t/ms, spkdata.a1.e.i, '|')
#%%
#%%
#datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/'
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/'

#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_YG = np.zeros([10,20,2])

#%%
for loop_num in range(400): # sys.argv[1]
    ee_ind = loop_num//20%10
    ie_ind  = loop_num%20 
    otr_ind = loop_num//200%2
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_YG[ee_ind, ie_ind, otr_ind] = spkdata.a1.e.rate_overall 

#%%
loop_num = 1
with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))
    

spkdata.a1.e.rate_overall

#%%
ee = np.linspace(0.7,1.1,10)
ie = np.linspace(2.4, 4.5, 20)
#%%
ee_num = 8
ie_num = 0
otr_num = 0
print(ee[ee_num], ie[ie_num])
#%%
plt.figure()
plt.plot(ie, np.log10(rate_mat_YG[ee_num,:,otr_num]), '*')
#%%
loop_num = otr_num*200 + ee_num*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.e.t = spkdata.a1.e.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 10*second
binforrate=5*ms
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spkdata.a1.e, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#spkrate2, centre_ind2, jump_size2, jump_dist2 = psa.get_rate_centre_jumpdist(spkdata.a2.e, starttime=starttime, endtime=endtime, binforrate=binforrate, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms)
#%%

spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)

#spkrate2 = psa.overlap_centreandspike(centre_ind2, spkrate2, show_trajectory = False)
#%%
anititle=''#'change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])

startstep = 0; endstep = 1000
frames = endstep - startstep
start_time = startstep
ani1 = psa.show_pattern(spkrate1[:,:,startstep:endstep], spkrate2=1, area_num = 1, frames = frames, start_time = start_time, anititle=anititle)
#%%
spkdata.a1.e.i.size/10/3969
spkdata.a1.param.__dict__
#%%
levy.fit_levy(jump_size1[:,0])
#%%
plt.figure()
plt.plot(spkdata.a1.e.t/ms, spkdata.a1.e.i, '|')

#%%
def foo(a,b):
    print(a)

#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/rk4/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_rk4 = np.zeros([10,20,2])

#%%
for loop_num in range(400): # sys.argv[1]
    ee_ind = loop_num//20%10
    ie_ind  = loop_num%20 
    otr_ind = loop_num//200%2
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_rk4[ee_ind, ie_ind, otr_ind] = spkdata.a1.ge.rate_overall 

#%%
loop_num = 200
with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))
    

spkdata.a1.ge.rate_overall

#%%
ee = np.linspace(0.7,1.1,10)
ie = np.linspace(2.4, 4.5, 20)
#%%
ee_num = 7
ie_num = 1
otr_num = 0
print(ee[ee_num], ie[ie_num])
#%%
plt.figure()
plt.plot(ie, np.log10(rate_mat_rk4[ee_num,:,otr_num]), '*')
#%%
loop_num = otr_num*200 + ee_num*20 + ie_num

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

startstep = 0; endstep = 1000
frames = endstep - startstep
start_time = startstep
ani1 = psa.show_pattern(spkrate1[:,:,startstep:endstep], spkrate2=1, area_num = 1, frames = frames, start_time = start_time, anititle=anititle)
#%%
spkdata.a1.e.i.size/10/3969
spkdata.a1.param.__dict__
#%%
levy.fit_levy(jump_size1[:,0])
#%%
plt.figure()
plt.plot(spkdata.a1.e.t/ms, spkdata.a1.e.i, '|')
#%%
from scipy.optimize import curve_fit
#%%
def power_law(x, a, b, c):
    return a*x**b + c
#%%
pars_1, cov_1 = curve_fit(power_law, ie[4:11],rate_mat_rk4[ee_num,4:11,otr_num])   
pars_2, cov_2 = curve_fit(power_law, ie[11:20],rate_mat_rk4[ee_num,11:20,otr_num])   

#%%
plt.figure()

y_1 = power_law(ie[4:11], *pars_1)
y_2 = power_law(ie[11:20], *pars_2)
#plt.figure()
plt.plot(ie[4:11], y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1[0],pars_1[1],pars_1[2]))
plt.plot(ie[11:20], y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_2[0],pars_2[1],pars_2[2]))
plt.plot(ie, rate_mat_rk4[ee_num,:,otr_num], '*')
plt.yscale('log')
plt.legend()
plt.title("Adam synapse model, yifan's params, e-e strength:%s"%('%.3fnS'%(ee[ee_num]*4)))
plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_AD_%.3f.svg'%(ee[ee_num]*4))
#%%
#datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/'
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/YG/low_ie/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_YG_lowie = np.zeros([10,10,2])

#%%
for loop_num in range(200): # sys.argv[1]
    ee_ind = loop_num//10%10
    ie_ind  = loop_num%10 
    otr_ind = loop_num//100%2
    try:
        with open(datapath+'data%d.file'%loop_num, 'rb') as file:
            #data = pickle.load(file)
            spkdata = mydata.mydata(pickle.load(file))
    except:        
        continue
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_YG_lowie[ee_ind, ie_ind, otr_ind] = spkdata.a1.ge.rate_overall 

#%%
loop_num = 183
with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))
    

#spkdata.a1.e.rate_overall
spkdata.a1.param.__dict__
#%%
ee = np.linspace(0.7,1.1,10)
ie = np.linspace(2.4, 4.5, 20)
ie_low = np.linspace(2.0, 3.38, 10)
#%%
ee_num = 7
ie_num = 0
otr_num = 1
print(ee[ee_num], ie[ie_num])
#%%
combine_YG = np.concatenate((rate_mat_YG_lowie[ee_num,:,otr_num], rate_mat_YG[ee_num,:,otr_num]))
combine_YG_ie = np.concatenate((ie_low, ie))
#%%
combine_YG = combine_YG[np.argsort(combine_YG_ie)]
combine_YG_ie = combine_YG_ie[np.argsort(combine_YG_ie)]
#%%
plt.figure()
#plt.plot(np.log10(ie), np.log10(rate_mat_YG_lowie[ee_num,:,otr_num]), '*')
plt.plot(combine_YG_ie, np.log10(combine_YG),'*')
#%%
combine_YG_ie_1 = combine_YG_ie[(combine_YG_ie<3.3) & (combine_YG_ie>2.2)]
combine_YG_1 = combine_YG[(combine_YG_ie<3.3) & (combine_YG_ie>2.2)]
combine_YG_ie_2 = combine_YG_ie[combine_YG_ie>3.45]
combine_YG_2 = combine_YG[combine_YG_ie>3.45]
#%%
#plt.plot(ie, rate_mat_YG_lowie[ee_num,:,otr_num], '*')
pars_1_yg, cov_1_yg = curve_fit(power_law, combine_YG_ie_1,combine_YG_1)   
pars_2_yg, cov_2_yg = curve_fit(power_law, combine_YG_ie_2,combine_YG_2)   

#%%
plt.figure()

y_1 = power_law(combine_YG_ie_1, *pars_1_yg)
y_2 = power_law(combine_YG_ie_2, *pars_2_yg)
#plt.figure()
plt.plot(combine_YG_ie_1, y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1_yg[0],pars_1_yg[1],pars_1_yg[2]))
plt.plot(combine_YG_ie_2, y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_2_yg[0],pars_2_yg[1],pars_2_yg[2]))
plt.plot(combine_YG_ie, combine_YG, '*')
plt.yscale('log')
plt.legend()
plt.title("Yifan synapse model, yifan's params, e-e strength:%s"%('%.3fnS'%(ee[ee_num]*4)))
plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_YG_%.3f.svg'%(ee[ee_num]*4))

#%%
loop_num = otr_num*100 + ee_num*10 + ie_num

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

startstep = 0; endstep = 1000
frames = endstep - startstep
start_time = startstep
ani1 = psa.show_pattern(spkrate1[:,:,startstep:endstep], spkrate2=1, area_num = 1, frames = frames, start_time = start_time, anititle=anititle)
#%%
spkdata.a1.e.i.size/10/3969
spkdata.a1.param.__dict__
#%%
levy.fit_levy(jump_size1[:,0])
#%%
plt.figure()
plt.plot(spkdata.a1.e.t/ms, spkdata.a1.e.i, '|')
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/set_planewave/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_pw = np.zeros([5,20])

#%%
for loop_num in range(100): # sys.argv[1]
    #ee_ind = loop_num//20%10
    ie_ind  = loop_num%20 
    ref_ind = loop_num//20%5
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_pw[ref_ind, ie_ind] = spkdata.a1.ge.rate_overall 

#%%
ref = np.linspace(5,5.5,5)
ie = np.linspace(2.4, 4.5, 20)
#%%
ref_num = 4
ie_num = 9
#otr_num = 0
print(ref[ref_num], ie[ie_num])
#%%
plt.figure()
plt.plot(ie, np.log10(rate_mat_pw[ref_num,:]), '*')
#%%
loop_num = ref_num*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.ge.t = spkdata.a1.ge.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 5*second
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
spkdata.a1.e.i.size/10/3969
spkdata.a1.param.__dict__
#%%
levy.fit_levy(jump_size1[:,0])
#%%
plt.figure()
plt.plot(spkdata.a1.e.t/ms, spkdata.a1.e.i, '|')
#%%
from scipy.optimize import curve_fit
#%%
def power_law(x, a, b, c):
    return a*x**b + c
#%%
pars_1, cov_1 = curve_fit(power_law, ie[4:10],rate_mat_rk4[ee_num,4:10,otr_num])   
pars_2, cov_2 = curve_fit(power_law, ie[11:20],rate_mat_rk4[ee_num,11:20,otr_num])   

#%%
plt.figure()

y_1 = power_law(ie[4:10], *pars_1)
y_2 = power_law(ie[11:20], *pars_2)
#plt.figure()
plt.plot(ie[4:10], y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1[0],pars_1[1],pars_1[2]))
plt.plot(ie[11:20], y_2, '-.',label='fit, y=a*x^b\na:%.2f b:%.2f'%(pars_2[0],pars_2[1]))
plt.plot(ie, rate_mat_rk4[ee_num,:,otr_num], '*')
plt.yscale('log')
plt.legend()
plt.title("Adam synapse model, yifan's params, e-e strength:%s"%('%.3fnS'%(ee[ee_num]*4)))
plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_AD_%.3f.svg'%(ee[ee_num]))

#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/set_planewave/repeat/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_pw_rep = np.zeros([10,20])

#%%
for loop_num in range(200): # sys.argv[1]
    #ee_ind = loop_num//20%10
    ie_ind  = loop_num%20 
    rep_ind = loop_num//20%10
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_pw_rep[rep_ind, ie_ind] = spkdata.a1.ge.rate_overall 

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
plt.errorbar(ie, rate_mat_pw_rep.mean(0), yerr=rate_mat_pw_rep.std(0), fmt='*')
#plt.errorbar(ie, rate_mat_pw_rep.mean(0), yerr=rate_mat_pw_rep.std(0), fmt='.')
plt.yscale('log')

#%%
from scipy.optimize import curve_fit
#%%
def power_law(x, a, b, c):
    return a*x**b + c
#%%
pars_1_pw, cov_1_pw = curve_fit(power_law, ie[4:10],rate_mat_pw_rep[:,4:10].mean(0))   
pars_2_pw, cov_2_pw = curve_fit(power_law, ie[11:20],rate_mat_pw_rep[:,11:20].mean(0))   

#%%
plt.figure()

y_1 = power_law(ie[4:10], *pars_1_pw)
y_2 = power_law(ie[11:20], *pars_2_pw)
#plt.figure()
plt.plot(ie[4:10], y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1_pw[0],pars_1_pw[1],pars_1_pw[2]))
plt.plot(ie[11:20], y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_2_pw[0],pars_2_pw[1],pars_2_pw[2]))
plt.errorbar(ie, rate_mat_pw_rep.mean(0), yerr=rate_mat_pw_rep.std(0), fmt='*')
plt.yscale('log')
plt.legend()
plt.title("Adam synapse model, yifan's params\n refrectory period: %.1fms, e-e strength:%.3fnS\ncan produce plane wave"%(5.5, spkdata.a1.param.mean_J_ee))#spkdata.a1.param.t_ref
plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_AD_long_tref_planewave_%.3f_2.svg'%(spkdata.a1.param.mean_J_ee))
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/tuneplanewave2/highie/'

loop_num = 202#otr_num*200 + ee_num*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.ge.t = spkdata.a1.ge.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 5*second
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
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/tuneplanewave2/'

loop_num = 183 #rep_ind*20 + ie_num

with open(datapath+'data%d.file'%loop_num, 'rb') as file:
    #data = pickle.load(file)
    spkdata = mydata.mydata(pickle.load(file))

#%%
spkdata.a1.ge.t = spkdata.a1.ge.t*0.1*ms
#spkdata.a2.e.t = spkdata.a2.e.t*0.1*ms

starttime = 0*second; endtime = 5*second
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
spkdata.a1.e.i.size/10/3969
spkdata.a1.param.__dict__
#%%
levy.fit_levy(jump_size1[:,0])
#%%
plt.figure()
plt.plot(spkdata.a1.e.t/ms, spkdata.a1.e.i, '|')
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/tuneplanewave2/set_planewave2/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/'
#datapath = '/import/headnode1/shni2598/brian2/data/attention/onearea_gz_param/gz_param_spkinput/squareinput/'
#%%
rate_mat_pw_rep_2 = np.zeros([10,26])

#%%
for loop_num in range(260): # sys.argv[1]
    #ee_ind = loop_num//20%10
    ie_ind  = loop_num%26
    rep_ind = loop_num//26%10
    with open(datapath+'data%d.file'%loop_num, 'rb') as file:
        #data = pickle.load(file)
        spkdata = mydata.mydata(pickle.load(file))
    
    #spkdata.a1.e.rate_overall = spkdata.a1.e.i.size/10/3969
    rate_mat_pw_rep_2[rep_ind, ie_ind] = spkdata.a1.ge.rate_overall 

#%%
#ref = np.linspace(5,5.5,5)
ie = np.linspace(3.2, 5.5, 26)
#%%
#ref_num = 4
rep_ind = 0
ie_num = 3
#otr_num = 0
print(ref[ref_num], ie[ie_num])
#%%
plt.figure()
plt.errorbar(ie, rate_mat_pw_rep_2.mean(0), yerr=rate_mat_pw_rep_2.std(0), fmt='*')
#plt.plot(ie, rate_mat_pw_rep_2[3,:], '*')

#plt.errorbar(ie, rate_mat_pw_rep.mean(0), yerr=rate_mat_pw_rep.std(0), fmt='.')
plt.yscale('log')

#%%
from scipy.optimize import curve_fit
#%%
def power_law1(x, a, b, c):
    return a*x**b + c
def power_law2(x, a, b):
    return a*x**b 
#%%
pars_1_pw_2, cov_1_pw_2 = curve_fit(power_law1, ie[3:13],rate_mat_pw_rep_2[:,3:13].mean(0))   
pars_2_pw_2, cov_2_pw_2 = curve_fit(power_law1, ie[15:],rate_mat_pw_rep_2[:,15:].mean(0),p0=np.array([6.8043e5,-10,0])) #[6.8043e5,-10,0]   
#%%
plt.figure()

y_1 = power_law1(ie[3:13], *pars_1_pw_2)
y_2 = power_law1(ie[15:], *pars_2_pw_2)
#plt.figure()
plt.plot(ie[3:13], y_1, '-', label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_1_pw_2[0],pars_1_pw_2[1],pars_1_pw_2[2]))
#plt.plot(ie[15:], y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f c:%.2f'%(pars_2_pw_2[0],pars_2_pw_2[1],pars_2_pw_2[2]))
plt.plot(ie[15:], y_2, '-.',label='fit, y=a*x^b+c\na:%.2f b:%.2f'%(pars_2_pw_2[0],pars_2_pw_2[1]))

plt.errorbar(ie, rate_mat_pw_rep_2.mean(0), yerr=rate_mat_pw_rep_2.std(0), fmt='*')
plt.yscale('log')
plt.legend()
plt.title("Adam synapse model, yifan's params\n e-i:%.5fuS, i-i:%.5fuS\ncan produce plane wave"%(spkdata.a1.param.w_ei,spkdata.a1.param.w_ii))
plt.xlabel('ie-ratio')
plt.ylabel('log(E neurons firing rate)')
#plt.savefig('rate-ie_AD_ei_ii_planewave.svg')
#%%
datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/AD/tuneplanewave2/set_planewave2/'

loop_num = 27 #rep_ind*20 + ie_num

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
