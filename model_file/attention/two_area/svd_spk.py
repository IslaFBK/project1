#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:45:23 2021

@author: shni2598
"""

#%%
import brian2.numpy_ as np
from brian2.numpy_.linalg import svd
#%%
np.random.poisson([2,10], (3,2))

#%%
lam = np.array([3, 9, 15])
fluc = np.random.rand(100)+0.5
#%%
np.random.poisson(lam*fluc[0])
#%%

Nspk = np.zeros([len(lam), len(fluc)])

for ii in range(len(fluc)):
    Nspk[:,ii] = np.random.poisson(lam*fluc[ii])

#%%
u, s, vh = np.linalg.svd(Nspk, full_matrices=True)

#%%
term1st = s[0]*np.dot(u[:,0].reshape(-1,1),vh[0,:].reshape(1,-1))
#%%
fig,ax=plt.subplots(2,1)
ax[0].imshow(Nspk,aspect='auto')
ax[1].imshow(term1st,aspect='auto')



#%%
Nspk_msb = Nspk - Nspk.mean(1).reshape(-1,1)

u_msb, s_msb, vh_msb = np.linalg.svd(Nspk_msb, full_matrices=True)

#%%
project = np.dot(u_msb[:,0], Nspk_msb)
#%%
np.correlate(project,fluc)
np.corrcoef(project,fluc)

#%%
Nspk_msb_norm = Nspk_msb/np.sqrt(np.sum(Nspk_msb**2,1)).reshape(-1,1)

u_msb_norm, s_msb_norm, vh_msb_norm = np.linalg.svd(Nspk_msb_norm, full_matrices=True)

