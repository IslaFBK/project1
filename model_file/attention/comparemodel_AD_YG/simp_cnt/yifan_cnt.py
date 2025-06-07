#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:22:22 2020

@author: shni2598
"""


from connection import pre_process
from connection import connect_interareas 
import connection as cn
#from analysis import post_analysis
import poisson_stimuli as psti
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import time

import brian2.numpy_ as np
import brian2.only
from brian2.only import *
#import pickle
import sys
import pickle
#%%
Ne = 3969; Ni = 1024
p_ee = 0.08; ie_ratio_ = 3.375
ijwd_yg = pre_process.get_ijwd(Ni=Ni)
ijwd_yg.p_ee = p_ee
ijwd_yg.w_ee_dist = 'lognormal'
ijwd_yg.hybrid = 0.4
ijwd_yg.cn_scale_weight = 2
ijwd_yg.cn_scale_wire = 2
ijwd_yg.iter_num=5

ijwd_yg.ie_ratio = ie_ratio_
scale_ee_1 = 1; scale_ei_1 = 1.
ijwd_yg.mean_J_ee = 4*10**-3*scale_ee_1 # usiemens
ijwd_yg.sigma_J_ee = 1.9*10**-3*scale_ee_1 # usiemens

ijwd_yg.w_ei = 5*10**(-3)*scale_ei_1 #usiemens
ijwd_yg.w_ii = 25*10**(-3)

ijwd_yg.change_dependent_para()
#param = {**ijwd_yg.__dict__}

ijwd_yg.generate_ijw()
ijwd_yg.generate_d_rand()
#%%
max_dist = 31.5*np.sqrt(2)
dist_bin = np.linspace(0, max_dist,11)

dist = cn.coordination.lattice_dist(ijwd_yg.lattice_ext,62,[0,0])

#%%
n_perbin_all = np.zeros(len(dist_bin)-1)

for i in range(len(dist_bin)-1):
    n_perbin_all[i] = np.sum((dist >= dist_bin[i]) & (dist < dist_bin[i+1])) 
#%%
n_perbin = np.zeros([ijwd_yg.Ne, len(dist_bin)-1])

for i in range(ijwd_yg.Ne):
    dist_i = ijwd_yg.dist_ee[ijwd_yg.i_ee == i]  
    for j in range(len(dist_bin)-1):        
        n_perbin[i, j] = np.sum((dist_i >= dist_bin[j]) & (dist_i < dist_bin[j+1]))
#%%
cnt_p = n_perbin.mean(0)/n_perbin_all
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p)
plt.yscale('log')   
#%%
def prob_cnt(lattice, dist_bin, N, dist_xy, i_xy):
    
    dist = cn.coordination.lattice_dist(lattice,62,[0,0])
    n_perbin_all = np.zeros(len(dist_bin)-1)

    for i in range(len(dist_bin)-1):
        n_perbin_all[i] = np.sum((dist >= dist_bin[i]) & (dist < dist_bin[i+1])) 
        
    n_perbin = np.zeros([N, len(dist_bin)-1])

    for i in range(N):
        dist_i = dist_xy[i_xy == i]  
        for j in range(len(dist_bin)-1):        
            n_perbin[i, j] = np.sum((dist_i >= dist_bin[j]) & (dist_i < dist_bin[j+1]))

    cnt_p = n_perbin.mean(0)/n_perbin_all
    return cnt_p
#%%
#/////////////// out prob
cnt_p_ee = prob_cnt(ijwd_yg.lattice_ext, dist_bin, ijwd_yg.Ne, ijwd_yg.dist_ee, ijwd_yg.i_ee)
#%%
xx = dist_bin[:-1]+0.5*(dist_bin[1]-dist_bin[0])
pk1 = 0.85; #pk2 = 1*0.82
tau1 = 8; #tau2 = 20*0.82
#xx = np.linspace(0,31,100)
#yy1 = pk1*np.exp(-xx/tau1)
yy = pk1*np.exp(-xx/(tau1))
plt.figure()
plt.plot(xx, cnt_p_ee)
plt.plot(xx, yy,'--')
plt.yscale('log')  

#%%
cnt_p_ie = prob_cnt(ijwd_yg.lattice_ext, dist_bin, ijwd_yg.Ni, ijwd_yg.dist_ie, ijwd_yg.i_ie)
#%%
xx = dist_bin[:-1]+0.5*(dist_bin[1]-dist_bin[0])
pk1 = 0.6; #pk2 = 1*0.82
tau1 = 20; #tau2 = 20*0.82
yy = pk1*np.exp(-xx/(tau1))
plt.figure()
plt.plot(xx, cnt_p_ie)
plt.yscale('log')  
#%%
cnt_p_ii = prob_cnt(ijwd_yg.lattice_inh, dist_bin, ijwd_yg.Ni, ijwd_yg.dist_ii, ijwd_yg.i_ii)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ii)
plt.yscale('log')
#%%
cnt_p_ei = prob_cnt(ijwd_yg.lattice_inh, dist_bin, ijwd_yg.Ne, ijwd_yg.dist_ei, ijwd_yg.i_ei)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ei)
plt.yscale('log')
#%%
#/////////////// in prob
cnt_p_ee_in = prob_cnt(ijwd_yg.lattice_ext, dist_bin, ijwd_yg.Ne, ijwd_yg.dist_ee, ijwd_yg.j_ee)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ee_in)
plt.yscale('log')   
#%%
cnt_p_ie_in = prob_cnt(ijwd_yg.lattice_inh, dist_bin, ijwd_yg.Ne, ijwd_yg.dist_ie, ijwd_yg.j_ie)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ie_in)
plt.yscale('log')  
#%%
cnt_p_ii_in = prob_cnt(ijwd_yg.lattice_inh, dist_bin, ijwd_yg.Ni, ijwd_yg.dist_ii, ijwd_yg.j_ii)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ii_in)
plt.yscale('log')
#%%
cnt_p_ei_in = prob_cnt(ijwd_yg.lattice_ext, dist_bin, ijwd_yg.Ni, ijwd_yg.dist_ei, ijwd_yg.j_ei)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ei_in)
plt.yscale('log')
#%%
dist2ilattice = cn.coordination.lattice_dist(ijwd_yg.lattice_inh,62,[0,0])

#%%
n_perbin_all_inh = np.zeros(len(dist_bin)-1)

for i in range(len(dist_bin)-1):
    n_perbin_all_inh[i] = np.sum((dist2ilattice >= dist_bin[i]) & (dist2ilattice < dist_bin[i+1])) 
#%%
from scipy.optimize import curve_fit

def exponential(x, a, b):
    return a*np.exp(x/b)
#%%
dist_x = dist_bin[:-1] + (dist_bin[1] - dist_bin[0])/2

pars_p, cov_p = curve_fit(exponential, dist_x,cnt_p_ei)   
#%%
exponential(dist_x, *pars_p )
#%%
fig, [ax1, ax2] = plt.subplots(1,2,  figsize=[10,5])
ax1.plot(dist_x, cnt_p_ei)
ax2.plot(dist_x, cnt_p_ei_in)
ax1.set_yscale('log',basey=np.e)
ax2.set_yscale('log',basey=np.e)
ax1.set_xlabel('dist');ax1.set_ylabel('p')
ax2.set_xlabel('dist');ax2.set_ylabel('p')
ax1.set_title('efferent(out) probability')
ax2.set_title('afferent(in) probability')
fig.suptitle('e-i connetion')
#%%
def findnumsyn(N, i_xy, dist_xy, radi):
    dist = np.zeros(N)
    for i in range(N):
        dist[i] = np.sum(dist_xy[i_xy == i] <= radi)
    return dist.mean()
#%%
syn_ee = findnumsyn(3969, ijwd_yg.i_ee, ijwd_yg.dist_ee, 9)
syn_ei = findnumsyn(3969, ijwd_yg.i_ei, ijwd_yg.dist_ei, 9)
syn_ie = findnumsyn(1024, ijwd_yg.i_ie, ijwd_yg.dist_ie, 9)
syn_ii = findnumsyn(1024, ijwd_yg.i_ii, ijwd_yg.dist_ii, 9)
#%%
dist = cn.coordination.lattice_dist(ijwd.e_lattice, 62, [0,0])
#%%
dist = cn.coordination.lattice_dist(ijwd.i_lattice, 62, [0,0])
#%%
tau_d = 7#20*0.86*1.15
p_peak = 0.88
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
syn_ee_sc = p_peak*np.sum(np.exp(-dist/tau_d)[dist<=9])
print(syn_ee_sc)
#%%
tau_d = 9#20*0.86*1.15
p_peak = 0.88
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
syn_ei_sc = p_peak*np.sum(np.exp(-dist/tau_d)[dist<=9])
print(syn_ei_sc)
#%%
tau_d = 20#20*0.86*1.15
p_peak = 0.52
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
syn_ie_sc = p_peak*np.sum(np.exp(-dist/tau_d)[dist<=9])
print(syn_ie_sc)
#%%
tau_d = 20#20*0.86*1.15
p_peak = 0.78
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
syn_ii_sc = p_peak*np.sum(np.exp(-dist/tau_d)[dist<=9])
print(syn_ii_sc)
#%%




tau_d = 8*0.8#20*0.86*1.15
p_peak = 0.8
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
syn_ee_sc = p_peak*np.sum(np.exp(-dist/tau_d)[dist<=9])
print(syn_ee_sc)
#%%
tau_d = 10*0.82#20*0.86*1.15
p_peak = 0.82
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
syn_ei_sc = p_peak*np.sum(np.exp(-dist/tau_d)[dist<=9])
print(syn_ei_sc)
#%%
tau_d = 20*0.86#20*0.86*1.15
p_peak = 0.58*0.79
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
syn_ie_sc = p_peak*np.sum(np.exp(-dist/tau_d)[dist<=9])
print(syn_ie_sc)

#%%
tau_d = 20*0.86#20*0.86*1.15
p_peak = 0.78
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
syn_ii_sc = p_peak*np.sum(np.exp(-dist/tau_d)[dist<=9])
print(syn_ii_sc)
