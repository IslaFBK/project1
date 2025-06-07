#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:56:58 2020

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
p_ee = 0.16; ie_ratio_ = 3.375
ijwd = pre_process.get_ijwd(Ni=Ni)
ijwd.p_ee = p_ee
ijwd.w_ee_dist = 'lognormal'
ijwd.hybrid = 0.4
ijwd.cn_scale_weight = 2
ijwd.cn_scale_wire = 2
ijwd.iter_num=5

ijwd.ie_ratio = ie_ratio_
scale_ee_1 = 1; scale_ei_1 = 1.
ijwd.mean_J_ee = 4*10**-3*scale_ee_1 # usiemens
ijwd.sigma_J_ee = 1.9*10**-3*scale_ee_1 # usiemens

ijwd.w_ei = 5*10**(-3)*scale_ei_1 #usiemens
ijwd.w_ii = 25*10**(-3)

ijwd.change_dependent_para()
#param = {**ijwd.__dict__}

ijwd.generate_ijw()
ijwd.generate_d_rand()
#%%
max_dist = 31.5*np.sqrt(2)
dist_bin = np.linspace(0, max_dist,11)

dist = cn.coordination.lattice_dist(ijwd.lattice_ext,62,[0,0])

#%%
n_perbin_all = np.zeros(len(dist_bin)-1)

for i in range(len(dist_bin)-1):
    n_perbin_all[i] = np.sum((dist >= dist_bin[i]) & (dist < dist_bin[i+1])) 
#%%
n_perbin = np.zeros([ijwd.Ne, len(dist_bin)-1])

for i in range(ijwd.Ne):
    dist_i = ijwd.dist_ee[ijwd.i_ee == i]  
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
cnt_p_ee = prob_cnt(ijwd.lattice_ext, dist_bin, ijwd.Ne, ijwd.dist_ee, ijwd.i_ee)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ee)
plt.yscale('log')   
#%%
cnt_p_ie = prob_cnt(ijwd.lattice_ext, dist_bin, ijwd.Ni, ijwd.dist_ie, ijwd.i_ie)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ie)
plt.yscale('log')  
#%%
cnt_p_ii = prob_cnt(ijwd.lattice_inh, dist_bin, ijwd.Ni, ijwd.dist_ii, ijwd.i_ii)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ii)
plt.yscale('log')
#%%
cnt_p_ei = prob_cnt(ijwd.lattice_inh, dist_bin, ijwd.Ne, ijwd.dist_ei, ijwd.i_ei)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ei)
plt.yscale('log')
#%%
#/////////////// in prob
cnt_p_ee_in = prob_cnt(ijwd.lattice_ext, dist_bin, ijwd.Ne, ijwd.dist_ee, ijwd.j_ee)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ee_in)
plt.yscale('log')   
#%%
cnt_p_ie_in = prob_cnt(ijwd.lattice_inh, dist_bin, ijwd.Ne, ijwd.dist_ie, ijwd.j_ie)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ie_in)
plt.yscale('log')  
#%%
cnt_p_ii_in = prob_cnt(ijwd.lattice_inh, dist_bin, ijwd.Ni, ijwd.dist_ii, ijwd.j_ii)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ii_in)
plt.yscale('log')
#%%
cnt_p_ei_in = prob_cnt(ijwd.lattice_ext, dist_bin, ijwd.Ni, ijwd.dist_ei, ijwd.j_ei)
#%%
plt.figure()
plt.plot(dist_bin[:-1], cnt_p_ei_in)
plt.yscale('log')
#%%
dist2ilattice = cn.coordination.lattice_dist(ijwd.lattice_inh,62,[0,0])

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