# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:07:50 2020

@author: nishe
"""

import numpy as np
from cfc_analysis.bandpassSig import bandpassSig 
from cfc_analysis.find_Hilbert import find_Hilbert 
from cfc_analysis.find_MI_cfc import find_MI_cfc 
#%%

class cfc(object):
    
    def __init__(self):
        """
        Fs: sampling frequency (Hz)
        phaseBand/ampBand: N*2 array-like, of which each row specifies the low and high 3-dB pass-band frequency of band-pass filter
        timeDim: scalar value, specifies which axis of input signal ('sigIn', in the format of numpy array) is time axis
        badChannels: 1-D array-like, specify the index of channels which will be ignored in the analysis
        filterOrder: scalar value, filter order of Butterworth filter
        discardTime_bandpassSig: scalar value(second), the period of time at the beginning and end of input signal that will be discarded during bandpass filtering
        discardTime_Hilbert: scalar value(second), the period of time at the beginning and end of input signal that will be discarded during Hilbert transform
        section_input_to_find_MI_cfc: 1-D array-like, the beginning and end time-index of signal that will be passed to function 'find_MI_cfc'
        optionMethod: (1 or 2 or 3 or 4), method of calculating phase-amplitude modulation index 
        optionSur: (1 or 2), method of surrogate 
        """
        self.Fs = None
        self.phaseBand = None
        self.ampBand = None
        self.timeDim = -1
        self.badChannels = None
        self.filterOrder = 8
        self.discardTime_bandpassSig = None
        self.discardTime_Hilbert = None
        self.section_input_to_find_MI_cfc = None # 1-D array with 2 elements, specifying the start and end index of input signal passed to "find_MI_cfc"  
        self.optionMethod = 1;
        self.optionSur = 2
    
    def find_cfc_from_rawsig(self, sigIn, return_Ampdist = False):# ,self.phaseBand, self.ampBand, self.timeDim, self.badChannels, self.filterOrder, self.discardTime_bandpassSig, self.discardTime_Hilbert, self.section_input_to_find_MI_cfc, self.optionMethod, self.optionSur):
        """
        sigIn: 1-D (time-series) array or 2-D (Num of Channels*time-series or time-series*Num of Channels) numpy array 
        time-series: Boolean, specifies if return the Amplitude distribution at different phase
        """
        must_define = ('Fs', 'phaseBand', 'ampBand')
        error = False
        error_message = ''
        for key in self.__dict__.keys():
            if (self.__dict__[key] is None) and  (key in must_define):
                error_message += 'Error: the value of "%s" is missing, please define "%s"\n' %(key,key)
                #print('Error: the value of "%s" is missing, please define "%s"' %(key,key))
                error = True
        if error: raise Exception(error_message)
        
        print('Start processing ...')
        for key in self.__dict__.keys():
            print('%s:'%key); print(self.__dict__[key])
                
        timeDim = np.arange(len(sigIn.shape))[self.timeDim]
        print('Band-pass filter signals ...')
        filsig_phase = bandpassSig(sigIn, self.phaseBand, self.Fs, timeDim, self.badChannels, self.filterOrder, self.discardTime_bandpassSig)
        filsig_amp = bandpassSig(sigIn, self.ampBand, self.Fs, timeDim, self.badChannels, self.filterOrder, self.discardTime_bandpassSig)
        timeDim += 1
        print('Do Hilbert transform on filtered signals ...')
        filsig_phase_hilb = find_Hilbert(filsig_phase, self.Fs, timeDim, self.discardTime_Hilbert)
        filsig_amp_hilb = find_Hilbert(filsig_amp, self.Fs, timeDim, self.discardTime_Hilbert)
        phaseIn = np.angle(filsig_phase_hilb)
        ampIn = np.abs(filsig_amp_hilb)
        
        print('Find Phase-Amplitude Modulation Index ...')
        if len(sigIn.shape) == 1:
            
            if self.section_input_to_find_MI_cfc is not None:
                phaseIn = phaseIn[:, self.section_input_to_find_MI_cfc[0]:self.section_input_to_find_MI_cfc[1]]
                ampIn = ampIn[:, self.section_input_to_find_MI_cfc[0]:self.section_input_to_find_MI_cfc[1]]

            MI_raw, MI_surr, meanBinAmp, MI_surr_detail = find_MI_cfc(phaseIn, ampIn, self.optionMethod, self.optionSur)
            if return_Ampdist:
                return MI_raw, MI_surr, meanBinAmp
            else:
                return MI_raw, MI_surr
            
        elif len(sigIn.shape) == 2:
            
            timeDim = np.arange(2)[self.timeDim]
            if timeDim == 0:
                phaseIn = np.swapaxes(phaseIn, 1, 2)
                ampIn = np.swapaxes(ampIn, 1, 2)
            
            if self.section_input_to_find_MI_cfc is not None:
                phaseIn = phaseIn[:,:,self.section_input_to_find_MI_cfc[0]:self.section_input_to_find_MI_cfc[1]]
                ampIn = ampIn[:,:,self.section_input_to_find_MI_cfc[0]:self.section_input_to_find_MI_cfc[1]]
                
            MI_raw_mat = np.zeros([phaseIn.shape[1], len(self.phaseBand), len(self.ampBand)])
            MI_surr_mat = np.zeros([phaseIn.shape[1], len(self.phaseBand), len(self.ampBand)])
            MI_raw_mat[:,:,:] = np.nan
            MI_surr_mat[:,:,:] = np.nan
            
            if return_Ampdist:
                Ampdist = np.zeros([phaseIn.shape[1], 20, len(self.phaseBand), len(self.ampBand)])
                Ampdist[:,:,:,:] = np.nan
                
            badChans = np.where(np.any(np.any(np.isnan(phaseIn), 2), 0))[0]
            
            for i in np.setdiff1d(np.arange(phaseIn.shape[1]), badChans):
                
                MI_raw, MI_surr, meanBinAmp, MI_surr_detail= find_MI_cfc(phaseIn[:, i, :], ampIn[:, i, :], self.optionMethod, self.optionSur)
                MI_raw_mat[i, :, :] = MI_raw
                MI_surr_mat[i, :, :] = MI_surr
                if return_Ampdist:
                    Ampdist[i] = meanBinAmp
            
            if return_Ampdist:
                return MI_raw_mat, MI_surr_mat, Ampdist
            else:
                return MI_raw_mat, MI_surr_mat
        
        else: pass
                
#%%

#import scipy.io as sio
##%%   
#data = sio.loadmat('AD_data0.mat')
##%%
#lfp1 = data['AD_data'][0,0]['LFP'][0,0]['lfp1']
#lfp = lfp1[[6,12],::10]
##%%
#        for key in self.__dict__.keys():
#            print(key);
#            print(self.__dict__[key]); 
#            print('\n')

#findcfc = cfc.cfc()
#Fs = 1000;
#timeDim = 0;
#phaseBand = np.arange(1,14.1,0.5)
#ampBand = np.arange(20,101,5) 
#phaseBandWid = 0.49 ;
#ampBandWid = 5 ;
#
#band1 = np.concatenate((phaseBand - phaseBandWid, ampBand - ampBandWid)).reshape(1,-1)
#band2 = np.concatenate((phaseBand + phaseBandWid, ampBand + ampBandWid)).reshape(1,-1)
#subBand = np.concatenate((band1,band2),0)
#subBand = subBand.T
##
###%%
#timeDim = -1;
#
#findcfc.Fs = Fs; 
#findcfc.phaseBand = subBand[:len(phaseBand)];
#findcfc.ampBand = subBand[len(phaseBand):]
#findcfc.find_cfc_from_rawsig(np.ones(1000))
#findcfc.section_input_to_find_MI_cfc = [4000,40000]
#MI_raw, MI_surr, meanBinAmp = findcfc.find_cfc_from_rawsig(lfp,return_Ampdist=True)
#%%
##%%
#fig, ax1 = plt.subplots(1,1, figsize=[8,6])
##im1 = ax1.imshow(np.flip(MI_raw.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid, phaseBand[-1]+phaseBandWid, ampBand[0]-ampBandWid, ampBand[-1]+ampBandWid])
#imcf = ax1.contourf(phaseBand, ampBand, MI_raw[0].T, 15)#, aspect='auto')
#imc = ax1.contour(phaseBand, ampBand, MI_raw[0].T, 15, colors='k', linewidths=0.6)#, aspect='auto')
#
#plt.colorbar(im1, ax=ax1)
##%%
#fig, ax1 = plt.subplots(1,1, figsize=[8,6])
##im1 = ax1.imshow(np.flip(MI_raw.T, 0), aspect='auto', extent=[phaseBand[0]-phaseBandWid, phaseBand[-1]+phaseBandWid, ampBand[0]-ampBandWid, ampBand[-1]+ampBandWid])
#imcf = ax1.contourf(phaseBand, ampBand, MI_surr[0].T, 15)#, aspect='auto')
#imc = ax1.contour(phaseBand, ampBand, MI_surr[0].T, 15, colors='k', linewidths=0.6)#, aspect='auto')
#
#plt.colorbar(im1, ax=ax1)
#%%
#findcfc = cfc.cfc()
#Fs = 1000;
#timeDim = 0;
#phaseBand = np.arange(1,14.1,0.5)
#ampBand = np.arange(20,101,5) 
#phaseBandWid = 0.49 ;
#ampBandWid = 5 ;
#
#band1 = np.concatenate((phaseBand - phaseBandWid, ampBand - ampBandWid)).reshape(1,-1)
#band2 = np.concatenate((phaseBand + phaseBandWid, ampBand + ampBandWid)).reshape(1,-1)
#subBand = np.concatenate((band1,band2),0)
#subBand = subBand.T
##
###%%
#timeDim = -1;
#
#findcfc.Fs = Fs; 
#findcfc.phaseBand = subBand[:len(phaseBand)];
#findcfc.ampBand = subBand[len(phaseBand):]
#findcfc.find_cfc_from_rawsig(np.ones(10000))
