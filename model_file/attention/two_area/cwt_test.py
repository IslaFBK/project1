#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:36:51 2021

@author: shni2598
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import frequency_analysis as fqa

#%%
cwt_t = 2*np.sin(2*np.pi*np.arange(10000)/1000)
cwt_t2 = 2*np.sin(4*np.pi*np.arange(10000)/1000)
cwt_t3 = np.zeros(10000)
cwt_t3[:5000] = cwt_t[:5000]
cwt_t3[5000:] = cwt_t2[5000:]
#%%
coef, freq = fqa.mycwt(cwt_t, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

fig, ax = plt.subplots(3,1,figsize=[9,9])
fig, ax[0], ax02 = fqa.plot_cwt(coef, freq, base = 10, fig=fig, ax=ax[0])

coef, freq = fqa.mycwt(cwt_t+cwt_t2, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

#fig, ax = plt.subplots(3,1,figsize=[9,9])
fig, ax[1], ax12 = fqa.plot_cwt(coef, freq, base = 10, fig=fig, ax=ax[1])

coef, freq = fqa.mycwt(cwt_t3, 'cmor1.5-1', 1e-3, scale = None,  method = 'fft', L1_norm = True)

#fig, ax = plt.subplots(3,1,figsize=[9,9])
fig, ax[2], ax22 = fqa.plot_cwt(coef, freq, base = 10, fig=fig, ax=ax[2])

#%%
plt.figure()
imcwt = plt.imshow(np.abs(coef), aspect='auto')
#%%
plt.figure()

plt.plot(cwt_t3)