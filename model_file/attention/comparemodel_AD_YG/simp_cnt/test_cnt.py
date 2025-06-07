#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:24:12 2020

@author: shni2598
"""
#%%
import matplotlib as mpl
#mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import connection as cn
import pre_process_sc
import pre_process
import firing_rate_analysis as fa
import pickle
import sys
import os
import matplotlib.pyplot as plt
import levy

import matplotlib.animation as animation
#%%
ijwd = pre_process_sc.get_ijwd()
#ijwd.Ne = 77*77; ijwd.Ni = 39*39
#ijwd.width = 76
#ijwd.w_ee_mean *= 2; ijwd.w_ei_mean *= 2; ijwd.w_ie_mean *= 2; ijwd.w_ii_mean *= 2; 
ijwd.decay_p_ee = 8 # decay constant of e to e connection probability as distance increases
ijwd.decay_p_ei = 10 # decay constant of e to i connection probability as distance increases
ijwd.decay_p_ie = 20 # decay constant of i to e connection probability as distance increases
ijwd.decay_p_ii = 20 # decay constant of i to i connection probability as distance increases

ijwd.mean_SynNumIn_ee = 320*5/8     ; # p = 0.08
ijwd.mean_SynNumIn_ei = 500     ; # p = 0.125
ijwd.mean_SynNumIn_ie = 200*5/8     ; # p = 0.2
ijwd.mean_SynNumIn_ii = 250     ; # p = 0.25

'''
Yifan:
ijwd.mean_SynNumIn_ee = 640     ; # p = 0.08
ijwd.mean_SynNumIn_ei = 800     ; # p = 0.125
ijwd.mean_SynNumIn_ie = 200     ; # p = 0.2
ijwd.mean_SynNumIn_ii = 400     ; # p = 0.25
'''


ijwd.generate_ijw()
ijwd.generate_d_rand()
#%%
ijwdyg = pre_process.get_ijwd()
ijwdyg.Ni = 1024; ijwdyg.Ne = 63*63 # number of i and e neurons
ijwdyg.p_ee = 0.05; ijwdyg.p_ie = 0.125 # connection probability
ijwdyg.p_ei = 0.125; ijwdyg.p_ii = 0.25
ijwdyg.iter_num = 1        
ijwdyg.hybrid = 0 # percent of lognormal part of the in- and out-degree 
ijwdyg.cn_scale_wire = 1 # common-neighbour factor for gennerating connectivity (synapses)
ijwdyg.cn_scale_weight = 1 
ijwdyg.change_dependent_para()        
ijwdyg.generate_ijw()


#%%
num = lambda x, tau, p: 2*np.pi*p*(-tau * np.e**(-x/tau)*x - tau**2*(np.e**(-x/tau) - 1))
percent = lambda x, tau: (-tau * np.e**(-x/tau)*x - tau**2*(np.e**(-x/tau) - 1))/tau**2 
#%%
num(10,20)
#%%
plt.figure()
plt.plot(ijwd.e_lattice[ijwd.j_ee[ijwd.i_ee==0]][:,0],ijwd.e_lattice[ijwd.j_ee[ijwd.i_ee==0]][:,1],'o')
#%%
def plot_cnt(i,j,src_neuron,trg_net):
    
    plt.figure()
    plt.plot(trg_net[j[i==src_neuron]][:,0], trg_net[j[i==src_neuron]][:,1],'o')
#%%
def plot_cnt_src(i,j,trg_neuron,src_net):
    
    plt.figure()
    plt.plot(src_net[i[j==trg_neuron]][:,0], src_net[i[j==trg_neuron]][:,1],'o')

#%%
plot_cnt(ijwd.i_ee, ijwd.j_ee, 130, ijwd.e_lattice)
plot_cnt(ijwdyg.i_ee, ijwdyg.j_ee, 130, ijwdyg.lattice_ext)

#%%
plot_cnt(ijwd.i_ie, ijwd.j_ie, 100, ijwd.e_lattice)
plot_cnt_src(ijwd.i_ie, ijwd.j_ie, 1000, ijwd.i_lattice)

#%%
plot_cnt(ijwd.i_ii, ijwd.j_ii, 100, ijwd.i_lattice)
plot_cnt(ijwdyg.i_ii, ijwdyg.j_ii, 100, ijwdyg.lattice_inh)
#%%
plot_cnt(ijwd.i_ie, ijwd.j_ie, 100, ijwd.e_lattice)
plot_cnt(ijwdyg.i_ie, ijwdyg.j_ie, 100, ijwd.e_lattice)
#%%
plot_cnt(ijwd.i_ei, ijwd.j_ei, 120, ijwd.i_lattice)
plot_cnt(ijwdyg.i_ei, ijwdyg.j_ei, 120, ijwdyg.lattice_inh)





