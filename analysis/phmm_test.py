#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:43:03 2021

@author: shni2598
"""

from phmm import PHMM, PHMM_d

#%%
theta = [[0.5, 0.5],
             [0.5, 0.5]]
delta = [1.0, 0.0]
lambdas = [5, 50]
#%%
# Random parameters
theta_2 = [[0.5, 0.5],
         [0.5, 0.5]]
delta_2 = [0.5, 0.5]
lambdas_2 = [5, 30]
# Initialization
# Initialization
#%%
h_1 = PHMM(delta, theta, lambdas)
seqs, states = zip(*[h_1.gen_seq() for _ in range(20)])
#%%
h_2 = PHMM(delta_2, theta_2, lambdas_2)
h_2.baum_welch(data_anly.onoff_asso.stim_noatt[0].mua)
#%%
print(h_2.transition_matrix())
print(h_2.lambdas)
v_2 = list(map(h_2.viterbi, data_anly.onoff_asso.stim_noatt[0].mua))
v_2 = np.array(v_2)
print(list(zip(states[0], v_2[0])))
#%%
print(h_2.log_likelihood(seqs))
#%%

plt.figure()
plt.plot(data_anly.onoff_asso.stim_noatt[0].mua[0])
plt.plot(v_2[0]*50)
#%%
h_2 = PHMM(delta_2, theta_2, lambdas_2)
h_2.baum_welch(data_anly.onoff_asso.stim_att[0].mua)
#%%
print(h_2.transition_matrix())
print(h_2.lambdas)
v_2 = list(map(h_2.viterbi, data_anly.onoff_asso.stim_att[0].mua))
v_2 = np.array(v_2)
print(list(zip(states[0], v_2[0])))
#%%
print(h_2.log_likelihood(seqs))
#%%
#%%

plt.figure()
plt.plot(data_anly.onoff_asso.stim_att[0].mua[0])
plt.plot(v_2[0]*100)

#%%
#%%
#%%
h_2 = PHMM(delta_2, theta_2, lambdas_2)
h_2.baum_welch(data_anly.onoff_sens.stim_noatt[0].mua)
#%%
print(h_2.transition_matrix())
print(h_2.lambdas)
v_2 = list(map(h_2.viterbi, data_anly.onoff_sens.stim_noatt[0].mua))
v_2 = np.array(v_2)
print(list(zip(states[0], v_2[0])))
#%%
print(h_2.log_likelihood(seqs))
#%%
#%%

plt.figure()
plt.plot(data_anly.onoff_sens.stim_noatt[0].mua[0])
plt.plot(v_2[0]*30)


