# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:05:53 2020

@author: nishe
"""

import numpy as np
from scipy import signal
#%%
def find_Hilbert(sigIn, Fs, timeDim=-1, discardTime=None): #badChannels=None,
    
    if len(sigIn.shape) >= 5:
        raise Exception('Error: wrong "sigIn" format')
    try: 
        timeDim = np.arange(len(sigIn.shape))[timeDim]
    except IndexError: print('Error: Invalid "timeDim" value')
    
#    if badChannels is None:
#        nanChans = np.any(np.isnan(sigIn), timeDim)
#        zeroChans =  np.all(sigIn==0, timeDim)
#        badChannels = np.where(np.logical_or(nanChans, zeroChans))[0]
#    else: badChannels = np.asarray(badChannels)
    
    if discardTime is None: 
        discardTime = np.min([0.2, 0.02*sigIn.shape[timeDim]/Fs]) # second
    
    discardTime_steps = int(np.round(discardTime*Fs))
    
    if sigIn.shape[timeDim] <= discardTime_steps*2 :
        raise Exception('The length of signals is smaller than discard time')
    
    outSection = np.array([discardTime_steps, sigIn.shape[timeDim]-discardTime_steps], dtype=int)

    ################# 
    hilbertOut = signal.hilbert(sigIn, axis=timeDim)
    
    if timeDim == 0:
        hilbertOut = hilbertOut[outSection[0]:outSection[1]]
    elif timeDim == 1:
        hilbertOut = hilbertOut[:, outSection[0]:outSection[1]]
    elif timeDim == 2:
        hilbertOut = hilbertOut[:, :, outSection[0]:outSection[1]]
    else:
        hilbertOut = hilbertOut[:, :, :, outSection[0]:outSection[1]]
    
    return hilbertOut
    
    
    
    
    
    
    
    
    
    


