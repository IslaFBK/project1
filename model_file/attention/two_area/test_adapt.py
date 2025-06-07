#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:24:28 2021

@author: shni2598
"""

import adapt_gaussian
#%%


adapt_value = adapt_gaussian.get_adaptation(base_amp = 16, \
    max_decrease = [14], sig=[6], position=[[0, 0]], n_side=64, width=64)
#%%

# plt.imshow(adapt_value.reshape(64,64))
# plt.colorbar()
#%%

adapt_gau = adapt_value.reshape(64,64)[32,:]
adapt_squ = np.ones(64)*16
adapt_squ[25:39] = 3.2
plt.figure()
plt.plot(adapt_gau)
plt.plot(adapt_squ)

