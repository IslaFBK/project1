#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:31:53 2021

@author: shni2598
"""

import numpy as np
#from brian2.only import *
import matplotlib.pyplot as plt

from connection import coordination
#%%
def get_adaptation(base_amp = 16, max_decrease = [13,13], sig=[7,7], position=[[0, 0],[-32, -32]], n_side=64, width=64):
    '''
    calculate value of adaptation(delta_gk) with uniform profile for each neuron
    base_amp: base-line amplitude of adaptation
    position: the coordinate of the modulation locations, modulation will decrease adaptation at corresponding locations.
    max_decrease: max decrease from 'base_amp' due to modulation
    sig: standard deviation of the gaussian profile of adaptation modulation
    '''
    hw = width/2
    step = width/n_side
    x = np.linspace(-hw + step/2, hw - step/2, n_side)
    y = np.linspace(hw - step/2, -hw + step/2, n_side)
    
    xx, yy = np.meshgrid(x,y)
    
    lattice = np.zeros([n_side*n_side, 2])
    lattice[:,0] = xx.flatten()
    lattice[:,1] = yy.flatten()

    n_stim = len(position)
    for i in range(n_stim):
        dist = coordination.lattice_dist(lattice, width, position[i])
        if i == 0:
            adapt_dec = np.zeros(len(dist))
        adapt_dec += max_decrease[i]*(dist <= sig[i])
    
    adapt = base_amp - adapt_dec

    return adapt
