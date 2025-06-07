#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 22:07:27 2021

@author: shni2598
"""

import frequency_analysis as fqa
from scipy import signal
#%%
f = 10;
t = np.arange(2000)/1000
#x = 3*np.sin(2*np.pi*f*t) + 5*np.sin(2*np.pi*2*f*t) + 2
x = 1*np.sin(2*np.pi*f*t) #+ 5*np.sin(2*np.pi*2*f*t) + 2

#%%
plt.figure()
plt.plot(x)

#%%
xcoef = np.fft.fft(x)

(np.abs(xcoef)**2).sum()/2000
(x**2).sum()
#%%
coef1, freq1 = fqa.myfft(x, Fs=1000, norm=True, power=False)
pow2, freq2 = fqa.myfft(x, Fs=1000, norm=True, power=True)

#%%
plt.figure()
plt.plot(freq1, np.abs(coef1))

plt.plot(freq2, np.abs(pow2))

#%%
(pow2*1).sum()
((x)**2*(1/1000)).sum()

#%%
plt.figure()

plt.plot(mua)
#%%
coef3, freq3 = fqa.myfft(mua, Fs=1000, norm=True, power=False)
pow3, freq3 = fqa.myfft(mua, Fs=1000, norm=True, power=True)
#%%
plt.figure()
plt.loglog(freq3[1:], np.abs(coef3)[1:])
plt.loglog(freq3[1:], np.abs(pow3)[1:])
#%%
(pow3*(1000/mua.shape[0])).sum()
((mua)**2*(1/1000)).sum()
#%%
freqwel, Pxx_den = signal.welch(mua-mua.mean(), fs=1000, nperseg=1024)
plt.figure()
plt.loglog(freqwel,Pxx_den)
#%%
plt.figure()
#plt.loglog(freq3[1:], np.abs(coef3)[1:])
plt.loglog(freq3[1:], np.abs(pow3)[1:])
#%%
(Pxx_den*(freqwel[1])).sum()

