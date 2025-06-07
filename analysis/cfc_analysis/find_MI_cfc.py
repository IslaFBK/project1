# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:16:53 2020

@author: nishe
"""
import numpy as np
from scipy import signal
from scipy import stats
#%%
def find_MI_cfc(sigPhase, sigAmp, optionMethod=1, optionSur=2):
        
    numSur = 200 ;
    numPhase = np.shape(sigPhase)[0] ;
    numAmp =  np.shape(sigAmp)[0] ;
    numBin = 20 ;
    MI_raw = np.zeros([numPhase, numAmp]) ;
    MI_surr = np.zeros(MI_raw.shape) ;
    meanBinAmp = np.zeros([numBin, numPhase, numAmp]) ;
    MI_surr_detail = np.zeros([numSur, numPhase, numAmp]) ;
    
    if optionMethod == 1:
        methodName = 'tort';
    elif optionMethod == 2:
        methodName = 'canolty' ;
    elif optionMethod == 3:
        methodName = 'ozkurt' ;
    elif optionMethod == 4:
        methodName = 'PLV' ;
    else: pass

    if optionMethod != 1:
        sigAmp = stats.zscore(sigAmp, axis=1, ddof=1) ;
    
    disp_progress = int(0.2*numPhase*numAmp)
    prgs = 0
    for phaseNum in range(numPhase):
        for ampNum in range(numAmp):        
            MI_raw[phaseNum,ampNum],MI_surr[phaseNum,ampNum], meanBinAmp[:,phaseNum,ampNum], \
                MI_surr_detail[:,phaseNum,ampNum] \
                = CalcMI(sigPhase[phaseNum,:],sigAmp[ampNum,:],methodName,optionSur) ;
            prgs += 1
            if prgs%disp_progress == 0:
                print('Complete percentage: %d%%' %(prgs/(numPhase*numAmp)*100))
    print('Complete percentage: 100% \nDone')
    
    return MI_raw, MI_surr, meanBinAmp, MI_surr_detail
 

#%%
def CalcMI(phaseIn, ampIn,approach, surrogates):
# function CalcMI calculates the modulation index using four methods.

    phase = phaseIn ;
    amp = ampIn ;
    
    # Variable to hold MI for all trials
    #MI_matrix_raw = [];
            
    if approach == 'tort':
        nbin = 20 ;
        MI, MeanAmp = calc_MI_tort(phase,amp,nbin);
            
    elif approach == 'ozkurt':
        MI = calc_MI_ozkurt(phase,amp);
        MeanAmp = np.zeros(20) ;
            
    elif approach == 'canolty':
        MI = calc_MI_canolty(phase,amp);
        MeanAmp = np.zeros(20) ;
            
    elif approach == 'PLV':
        MI = calc_MI_PLV(phase,amp);
        MeanAmp = np.zeros(20) ;    
    else: pass
    
    # Add the MI value to all other all other values
    MI_matrix_raw = MI;
    #MI_matrix_surr = [] ;
                
    if surrogates==1: # random phase         
        # Variable to surrogate MI
        #MI_surr = [];
        numsurrogate = 200 ;
        MI_surr = np.empty(numsurrogate) #[];

        # For each surrogate (currently hard-coded for 200, could be changed)...
        for surr in range(numsurrogate):
            # Get 2 random trial numbers
            #trial_num = randperm(length(phaseIn),2);
            
            # Extract phase and amp info using hilbert transform
            # for different trials & shuffle phase
            phase = np.random.permutation(phaseIn); # getting the phase
            amp = ampIn;
            
            # Switch PAC approach based on user input
            if approach == 'tort':
                nbin = 20 ;
                MI, _ = calc_MI_tort(phase,amp,nbin);
                    
            elif approach == 'ozkurt':
                MI = calc_MI_ozkurt(phase,amp);
                    
            elif approach == 'canolty':
                MI = calc_MI_canolty(phase,amp);
                    
            elif approach == 'PLV':
                MI = calc_MI_PLV(phase,amp);
            else: pass
            
            # Add this value to all other all other values
            MI_surr[surr] = MI;
                
        # Subtract the mean of the surrogaates from the actual PAC
        # value and add this to the surrogate matrix
        MI_surr_normalised = MI_matrix_raw - np.mean(MI_surr);
        MI_matrix_surr = MI_surr_normalised;
     
    elif surrogates==2: #random shifted phase, edited from Canolty et al.
         
         # Variable to surrogate MI
        
        numsurrogate = 200 ;
        MI_surr = np.empty(numsurrogate) #[];

        # skip = ceil(length(phase).*rand(numsurrogate,1)); 
    
        numpoints = phase.shape[0] ;
        minskip = 1000 ;
        maxskip = numpoints - minskip ;
        if maxskip <= minskip:
            raise Exception("Error: Input signal doesn't have enough length to perform surrogate")
        #skip=ceil(numpoints.*rand(numsurrogate*2,1));  
        skip = np.random.randint(minskip, maxskip, numsurrogate)
        #skip((skip>maxskip))=[];
        #skip((skip<minskip))=[];
        #skip=skip(1:numsurrogate,1);
    
        # For each surrogate (surrently hard-coded for 200, could be changed)...
        for surr in range(numsurrogate):
            # Get 2 random trial numbers
            #amp=[ampIn(skip(surr):end), ampIn(1:skip(surr)-1)];
            amp = np.concatenate((ampIn[skip[surr]:], ampIn[:skip[surr]]))
            phase = phaseIn;
            
            # Switch PAC approach based on user input
            if approach == 'tort':
                nbin = 20 ;
                MI, _ = calc_MI_tort(phase,amp,nbin);
                    
            elif approach == 'ozkurt':
                MI = calc_MI_ozkurt(phase,amp);
                    
            elif approach == 'canolty':
                MI = calc_MI_canolty(phase,amp);
                    
            elif approach == 'PLV':
                MI = calc_MI_PLV(phase,amp);
            else: pass
            
            # Add this value to all other all other values
            MI_surr[surr] = MI;
        
        # Subtract the mean of the surrogaates from the actual PAC
        # value and add this to the surrogate matrix
        MI_surr_normalised = MI_matrix_raw - np.mean(MI_surr);
        MI_matrix_surr = MI_surr_normalised;
    
    return MI_matrix_raw, MI_matrix_surr, MeanAmp, MI_surr

#%%
def calc_MI_tort(Phase, Amp, nbin):
    '''
    Apply Tort et al (2010) approach)
    '''
    #nbin=18; % % we are breaking 0-360o in 18 bins, ie, each bin has 20o
    
#    position = np.zeros(nbin); # this variable will get the beginning (not the center) of each bin
#    # (in rads)
#    winsize = 2*np.pi/nbin;
#    for j in range(nbin):
#        position[j] = -np.pi+j*winsize;    
    position = np.linspace(-np.pi, np.pi, nbin+1)
    
   # now we compute the mean amplitude in each phase:
    MeanAmp = np.zeros(nbin);
    for j in range(nbin):
        #I = Amp[Phase < position[j+1] & Phase >= position[j]];
        #MeanAmp(j)=mean(Amp(I));
        MeanAmp[j] = np.mean(Amp[(Phase < position[j+1]) & (Phase >= position[j])]);
    
#    
#    % The center of each bin (for plotting purposes) is
#    % position+winsize/2
#    
#    % % Plot the result to see if there's any amplitude modulation
#    % if strcmp(diag, 'yes')
#    %     bar(10:20:720,[MeanAmp,MeanAmp]/sum(MeanAmp),'phase_freq')
#    %     xlim([0 720])
#    %     set(gca,'xtick',0:360:720)
#    %     xlabel('Phase (Deg)')
#    %     ylabel('Amplitude')
#    % end
    
#    % Quantify the amount of amp modulation by means of a
#    % normalized entropy index (Tort et al PNAS 2008):
    
    #MI=(log(nbin)-(-sum((MeanAmp/sum(MeanAmp)).*log((MeanAmp/sum(MeanAmp))))))/log(nbin);
    MI = 1 - ((np.sum(MeanAmp/np.sum(MeanAmp) * np.log(MeanAmp/np.sum(MeanAmp)))*(-1))/np.log(nbin))

    return MI, MeanAmp 

#%%
def calc_MI_ozkurt(Phase, Amp):
    # Apply the algorithm from Ozkurt et al., (2011)
    N = Amp.shape[0];
    z = Amp*np.exp(1j*Phase); # Get complex valued signal
    MI = (1/np.sqrt(N)) * np.abs(np.mean(z)) / np.sqrt(np.mean(Amp*Amp)); # Normalise
    return MI

def calc_MI_PLV(Phase, Amp):
    # Apply PLV algorith, from Cohen et al., (2008)
    amp_phase = np.angle(signal.hilbert(signal.detrend(Amp))); # Phase of amplitude envelope
    MI = np.abs(np.mean(np.exp(1j*(Phase-amp_phase))));
    return MI

def calc_MI_canolty(Phase, Amp):
    # Apply MVL algorith, from Canolty et al., (2006)
    z = Amp*np.exp(1j*Phase); # Get complex valued signal
    MI = np.abs(np.mean(z));
    return MI









