#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:51:42 2020

@author: shni2598
"""

import brian2.numpy_ as np
#from brian2.only import *
import matplotlib.pyplot as plt

from connection import coordination
#%%
def input_spkrate(maxrate = [800,800], sig=[6,6], position=[[-32, -32],[0, 0]], sti_type='Gaussian', n_side=64, width=64):
    '''
    generate the firing rate for the input poisson spike, the poisson rate of each input
    poisson spike has gaussian shape profile across the network
    position: the coordinate of the two peaks of two input stimuli
    maxrate: max rate of poisson input
    sig: standard deviation of the gaussian profile of input stimuli
    '''
    # #width = 62
    # hw = width/2
    # #n_side = 63
        
    # x = np.linspace(-hw,hw,n_side) #+ centre[0]
    # y = np.linspace(hw,-hw,n_side) #+ centre[1]
    hw = width/2
    step = width/n_side
    x = np.linspace(-hw + step/2, hw - step/2, n_side)
    y = np.linspace(hw - step/2, -hw + step/2, n_side)
    
    xx, yy = np.meshgrid(x,y)
    
    lattice = np.zeros([n_side*n_side, 2])
    lattice[:,0] = xx.flatten()
    lattice[:,1] = yy.flatten()
      
    #sti1 = [-31.5, -31.5]
    #sti2 = [0, 0]
    # number of stimulus
    n_stim = len(position)
    if sti_type == 'Gaussian':
        for i in range(n_stim):
            dist_sti = coordination.lattice_dist(lattice, width, position[i])
            if i == 0:
                rate_sti = np.zeros(len(dist_sti))
            rate_sti += maxrate[i]*np.exp((-0.5)*(dist_sti/sig[i])**2)
    else:
        for i in range(n_stim):
            dist_sti = coordination.lattice_dist(lattice, width, position[i])
            if i==0:
                rate_sti = np.zeros(len(dist_sti))
            # Uniform rate within sig radius, 0 outside
            rate_sti += maxrate[i] * (dist_sti <= sig[i])

#    dist_sti1 = coordination.lattice_dist(lattice, width, position[0])
#    dist_sti2 = coordination.lattice_dist(lattice, width, position[1])
#    
#    sig1 = sig2 = sig#(width+1)*(0.6/(2*np.pi))
#    maxr1 = maxr2 = maxrate
#    rsti1 = maxr1*np.exp((-0.5)*(dist_sti1/sig1)**2)
#    rsti2 = maxr2*np.exp((-0.5)*(dist_sti2/sig2)**2)
    
    return rate_sti #rsti1+rsti2
#%%
#a=input_spkrate()
#rate = input_spkrate(maxrate=[600,800], sig=[2,6])
#rate = rate.reshape(63,63)
##%%
#plt.figure()
#plt.imshow(rate)
#plt.colorbar()

