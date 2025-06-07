# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:24:22 2020

@author: nishe
"""

import numpy as np
from scipy import signal
#import pdb
#%%
def bandpassSig(sigIn, subBand, Fs, timeDim=-1, badChannels=None, filterOrder=8, discardTime=None):
    
    if len(sigIn.shape) == 1:
        if not (timeDim==0 or timeDim==-1):
            raise Exception('Error: Invalid "timeDim" value for 1-D "sigIn" array, "timeDim" must be either "0" or "-1" for 1-D "sigIn" array')
        sigIn = sigIn.reshape(1,-1) # reshape to '1*time' 2-D array for consistency of following processing
        timeDim = 1
    elif len(sigIn.shape) == 2:
        try: timeDim = np.arange(2)[timeDim]
        except IndexError: print('Error: Invalid "timeDim" value for 2-D sigIn array')
    else:
        print('Error: "sigIn" must be either 1-D or 2-D array')
               
    if badChannels is None:
        nanChans = np.any(np.isnan(sigIn), timeDim)
        zeroChans =  np.all(sigIn==0, timeDim)
        badChannels = np.where(np.logical_or(nanChans, zeroChans))[0]
    else: badChannels = np.asarray(badChannels)
    
    if discardTime is None: 
        discardTime = np.min([1, 0.1*sigIn.shape[timeDim]/Fs]) # second
    
    discardTime_steps = int(np.round(discardTime*Fs))
    
    if sigIn.shape[timeDim] <= discardTime_steps*2 :
        raise Exception('The length of signals is smaller than discard time')
    
    subBand = np.asarray(subBand)
    #outSection = np.arange(discardTime_steps, sigIn.shape[timeDim]-discardTime_steps, dtype=int)
    outSection = np.array([discardTime_steps, sigIn.shape[timeDim]-discardTime_steps], dtype=int)
    
    bandpassSig = np.zeros([subBand.shape[0],*sigIn.shape])
    
    bandpassSig[:,:,:] = np.nan
    
    if timeDim == 1:
        goodChannels = np.setdiff1d(np.arange(sigIn.shape[0]), badChannels)
    elif timeDim == 0:
        goodChannels = np.setdiff1d(np.arange(sigIn.shape[1]), badChannels)
    else: pass
    
    for bandNum in range(subBand.shape[0]):
        
        Wn = subBand[bandNum]/(Fs/2)
        sos = signal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')
        if timeDim == 1:
            filtout = signal.sosfiltfilt(sos, sigIn[goodChannels, :], axis=timeDim)
            bandpassSig[bandNum, goodChannels, :] = filtout
        else:
            filtout = signal.sosfiltfilt(sos, sigIn[:, goodChannels], axis=timeDim)
            #bandpassSig[bandNum, :, goodChannels] = filtout
            bandpassSig[bandNum][:, goodChannels] = filtout # don't use "bandpassSig[bandNum, :, goodChannels]" !
    
    if timeDim == 1:
        bandpassSig = bandpassSig[:,:,outSection[0]:outSection[1]]
        if bandpassSig.shape[1] == 1:
            bandpassSig = bandpassSig.reshape(bandpassSig.shape[0], -1) # reshape to '(size of bands)*time' 2-D array if sigIn was 1-D array originally
    else:
        bandpassSig = bandpassSig[:,outSection[0]:outSection[1],:]
        
    return bandpassSig

#%%






