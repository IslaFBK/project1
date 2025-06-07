#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:36:01 2021

@author: shni2598
"""
import poisson_stimuli as psti
#%%
hw = 31.5
x = -30.1; y = 31
print(((x - y + hw)%(2*hw) - hw))
#%%
hw = 31
x = -30.1; y = 31
print(((x - y + hw)%(2*hw+1) - hw))
#%%
hw = 31.5
step = 63/63
xx = np.linspace(-hw + step/2, hw - step/2, 32)
#%%
hw = 31.5
x = xx[0]; y = xx[-1]
print(((x - y + hw)%(2*hw) - hw))
#%%
hw = 31
x = xx[0]; y = xx[20]
print(((x - y + hw)%(2*hw+1) - hw))
#%%
import coordination
#%%
e_lattice = coordination.makelattice(63, 63, [0,0])
i_lattice = coordination.makelattice(32, 63, [0,0])
#%%
plt.figure()
plt.plot(e_lattice[:,1], e_lattice[:,0],'o',)
plt.plot(i_lattice[:,1], i_lattice[:,0],'o',)

#%%
stim_e = psti.input_spkrate(maxrate = [200], sig=[6], position=[[0, 0]], n_side=63, width=63)#*Hz
stim_i = psti.input_spkrate(maxrate = [200], sig=[6], position=[[0, 0]], n_side=32, width=63)#*Hz

#%%
plt.figure()
plt.imshow(stim_e.reshape(63,63))
#%%
plt.figure()
plt.imshow(stim_i.reshape(32,32))
#%%
plt.figure()
plt.plot(np.arange(5),'--o')



