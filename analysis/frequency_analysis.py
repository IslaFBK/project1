#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:51:24 2020

@author: shni2598
"""

import numpy as np
import pywt
from scipy import signal as spsignal
import matplotlib.pyplot as plt
import scipy.signal.windows as sp_windows
import mydata
#import pdb
#%%
def myfft(data, Fs, norm=True, power=False):
    '''
    Do discret fourier transform for real-value input signal.
    Negative frequency components are discarded.
    input:
    data: real-value numpy array. 
    Fs: sampling frequency (Hz)
    norm: whether to normalize coef, i.e., |coef| = the magnitude of corresponding frequency component in time domain.
    power: if return power spectrum, i.e., coef is power density of corresponding frequency
    output:
    coef: fourier coefficients
    freq: corresponding frequency of "coef"
    '''
    coef = np.fft.fft(data)
    if norm and not power:
        coef = coef/coef.shape[0]
    if len(data)%2 == 0:
        freq = np.arange(len(data)/2 + 1)/(len(data))*Fs
        coef = coef[0:len(freq)]
    else:
        freq = np.arange(len(data)//2 + 1)/(len(data))*Fs
        coef = coef[0:len(freq)]
    
    if power:
        T = data.shape[0]/Fs
        coef = 2/Fs**2*np.abs(coef)**2/T
        coef[0] /= 2
    else:
        coef *= 2
    
    return coef, freq

#%%
def find_peakF(coef, freq, lwin):
    dF = freq[1] - freq[0]
    #Fwin = 0.3
    #lwin = 3#int(Fwin/dF)
    win = np.ones(lwin)/lwin
    coef_avg = np.convolve(np.abs(coef[1:]), win, mode='same')
    peakF = freq[1:][coef_avg.argmax()]
    return peakF

#%%
def plot_fft(freq, coef, freq_max1=20, freq_max2 = 200, fig=None, ax=None, show_theta=False, label=''):
    if fig is None:
        fig, ax = plt.subplots(2,2,figsize=[9,9])
       
    #peakF = find_peakF(coef, freq, 3)
    
    #freq_max1 = 20
    ind_len = freq[freq<freq_max1].shape[0] # int(20/(fs/2)data_fft*(len(data_fft)/2)) + 1
    ax[0,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_linear')
    ax[0,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_loglog')
    
    #freq_max2 = 150
    ind_len = freq[freq<freq_max2].shape[0] # int(20/(fs/2)*(len(data_fft)/2)) + 1
    ax[1,0].plot(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_linear')
    ax[1,1].loglog(freq[1:ind_len], np.abs(coef[1:ind_len]),label=label+'_loglog')
    
    if show_theta:
        clr = plt.rcParams['axes.prop_cycle'].by_key()['color']
        x, y = np.zeros([2,3]), np.zeros([2,3])
        x[:,0] = 3; x[:,1] = 5; x[:,2] = 8; 
        y[0,:] = 0; y[1,:] = np.max(np.abs(coef[1:ind_len]))
        ax[0,0].plot(x, y, c=clr[1])
        ax[0,1].loglog(x, y, c=clr[1])
    
    for i in range(2):
        for j in range(2):
            ax[i,j].legend()
    
    return fig, ax, #peakF#, coef, freq

#%%
def mycwt(signal, wavelet_name, sampling_period, scale = None,  method = 'fft', L1_norm = True):
    '''
    do continuous wavelet transform
    input:
    signal: 1-D numpy array, time series signal
    scale: 1-D numpy array, sacle of each wavelet kernel 
    wavelet_name: the name of wavelet to be used, e.g. 'cmor1.5-1'
    sampling_period: (second) sampling_period of input signal
    L1-norm: whether to normalise the output coefficient by using L1-norm. L1-norm is preferred since under the L1-norm, 
    if different frequency components in the input signal have the same magnititude, their corresponding output wavelet coefficients also
    have the same magnitude.
    output:
    coef: 2-D numpy array, output coefficents. Its number of row is equal to the length of "scale".
    freq: 1-D numpy array, frequency of corresponding scale. Its length is equal to the number of row of "coef".
    freq = 1/scale * (central-frequency of wavelet) * (1/sampling_period)
    '''
    if scale is None:
        freq_range = [0.5,250]
        maxscale = int(np.ceil(np.log2((1/sampling_period)/freq_range[0])*10))
        minscale = int(np.floor(np.log2((1/sampling_period)/freq_range[1])*10))
        scale = 2**(np.arange(minscale, maxscale + 1)/10)
    coef, freq = pywt.cwt(signal, scales=scale, wavelet=wavelet_name, sampling_period=sampling_period, method=method)
    coef *= 2 # *2 to get correct magnitude 
    if L1_norm:
        coef = (1/np.sqrt(scale)).reshape(-1,1) * coef
    
    return coef, freq

def plot_cwt(coef, freq, base = 10, fig=None, ax=None, colorbar=True):
    
    fix_bound = (freq[0]/freq[1])**0.5#2**(0.5*1/10)
    if fig is None:
        fig = plt.figure()
        ax = plt.subplot(111)#,label='cwt')
    imcwt = ax.imshow(np.abs(coef), aspect='auto')
    ax.yaxis.set_visible(False)
    if colorbar: plt.colorbar(imcwt, ax=ax)
    
    #ax.get_position()
    ax2 = fig.add_axes(ax.get_position(),frame_on=False)
    ax2.set_ylim([freq[-1]/fix_bound,freq[0]*fix_bound])
    ax2.xaxis.set_visible(False)
    
    # ax2 = fig.add_subplot(111,label='yaxis')
    # #ax2.yaxis.set_ticks(freq_wvt)
    # ax2.set_ylim([freq_wvt[-1]/(2**(1/10)),freq_wvt[0]*(2**(1/10))])
    # ax2.xaxis.set_visible(False)
    
    plt.yscale('log',base=base)
    return fig, ax, ax2
#%%
class taper_coherence_multisignal:
    
    def __init__(self):
        '''
        nw = T*hW_real = N*hW_digi = N*(hW_real/Fs) 
        nw : (or TW) time-half-bandwidth product
        T: length of signal (in second)
        hW_real: half-bandwith (in Hz)
        N: length of signal (number, dimensionless)
        hW_digi: half-bandwith (digital freqency, dimensionless)
        Fs: sampling frequency (Hz)
        '''
        
        self.hW_real = 10 # hz 3
        self.Fs = 1000 # hz
        self.dura = np.array([[]]) # 2D arrat segmentation of data to be analyzed
        #T = 1000; # ms
        #hb = 10 # hz
    
    def get_coherence(self, data_1, data_2, return_singalspectrum = False):
        
        #dura_persample = int(round((self.dura[0,1] - self.dura[0,0])/(1/self.Fs*1000)))
        
        dura = np.round(self.dura*self.Fs/1000).astype(int)
        dura_persample = dura[0,1] - dura[0,0]
    
        nw = int(round(dura_persample*self.hW_real/self.Fs))
        ntap = int(np.floor(2*nw - 1))
        win_dpss = sp_windows.dpss(dura_persample, nw, ntap, sym=False)

        n_data = dura.shape[0]
        if dura_persample%2 == 0: fft_len = int(round(dura_persample/2)) + 1
        else: fft_len = int(round((dura_persample+1)/2))


        
        if return_singalspectrum:
            s12_all = np.zeros([n_data, fft_len], dtype=complex)
            s1_all = np.zeros([n_data, fft_len], dtype=float)
            s2_all = np.zeros([n_data, fft_len], dtype=float)
        else:
            s12 = np.zeros(fft_len, dtype=complex)
            s1 = np.zeros(fft_len, dtype=float)
            s2 = np.zeros(fft_len, dtype=float)
        
        
        for dura_i in dura:
            #print(d1.shape,d2.shape)
            tap_data_1 = win_dpss*data_1[dura_i[0]:dura_i[1]]
            tap_data_2 = win_dpss*data_2[dura_i[0]:dura_i[1]]
            
            coef_1 = np.fft.rfft(tap_data_1, axis = -1)
            coef_2 = np.fft.rfft(tap_data_2, axis = -1)
            
            if return_singalspectrum:
                s12_all[dura_i] = np.mean(coef_1 * np.conjugate(coef_2), 0)
                s1_all[dura_i] = np.mean(np.abs(coef_1)**2, 0)
                s2_all[dura_i] = np.mean(np.abs(coef_2)**2, 0)
            else:
                s12 += np.mean(coef_1 * np.conjugate(coef_2), 0)/n_data
                s1 += np.mean(np.abs(coef_1)**2, 0)/n_data
                s2 += np.mean(np.abs(coef_2)**2, 0)/n_data
        
        if return_singalspectrum:
            s12 = s12_all.mean(0)
            s1 = s1_all.mean(0)
            s2 = s2_all.mean(0)
        
        cohe = np.abs(s12/np.sqrt(s1*s2))
        phase = np.angle(s12/np.sqrt(s1*s2))

        if return_singalspectrum:
            R = mydata.mydata()
            R.s1 = s1
            R.s2 = s2
            R.s12 = s12
            R.s12_all = s12_all
            R.s1_all = s1_all
            R.s2_all = s2_all
            R.cohe = cohe
            R.phase = phase
            
            return R
        else:
            R = mydata.mydata()
            R.s1 = s1
            R.s2 = s2
            R.s12 = s12
            R.cohe = cohe
            R.phase = phase
            
            return R
#%%
class samefreq_coupling:
    
    def __init__(self):    
        self.window = 30; 
        self.sample_interval = 1
        self.dt = 1 #ms
        
    def get_xcorr(self, data1, data2, band, mode = 'cov'):
        
        sample_interval = int(np.round(self.sample_interval/self.dt))
        window_step = int(np.round(self.window/self.dt))
        Fs = int(round(1/(self.dt*1e-3)))
        data1_filt, data2_filt, phase_diff_all = get_filt_phaseDiff(data1, data2, band, Fs = Fs, filterOrder = 8)
    
        hf_win = int(np.round(window_step/2))

        start_time = 0 # int(np.round(start_time/dt))
        end_time = data1_filt.shape[0] # vint(np.round(end_time/dt))

        xcorr_len = 2*window_step - 1
        sample_t = np.arange(start_time, end_time-window_step+1, sample_interval)
        xcorr = np.zeros([xcorr_len, sample_t.shape[0]])
        phase_diff = np.zeros(sample_t.shape)

        for ind, t in enumerate(sample_t):
            d1_seg = data1_filt[t:t+window_step] - data1_filt[t:t+window_step].mean() 
            d2_seg = data2_filt[t:t+window_step] - data2_filt[t:t+window_step].mean() 
            
            if mode == 'cov':
                xcorr_ = np.correlate(d1_seg, d2_seg, 'full')/window_step#/np.sqrt(np.sum(d1_seg**2) * np.sum(d2_seg**2))
            elif mode == 'corr':
                xcorr_ = np.correlate(d1_seg, d2_seg, 'full')/np.sqrt(np.sum(d1_seg**2) * np.sum(d2_seg**2))
            else: raise Exception("mode should be either 'cov' or 'corr'!")    
            #xcorr_ d1[t:t+window_step]
            xcorr[:,ind] = xcorr_ #np.correlate(d1[t:t+window_step], d2[t:t+window_step], 'full')
            phase_diff[ind] = phase_diff_all[t+hf_win]
        
        return xcorr, phase_diff
    
def get_filt_phaseDiff(data1, data2, band, Fs = 1000, filterOrder = 8):
    
    #Band = np.array([90,100])
    #Fs = 1000
    #Wn = subBand[bandNum]/(Fs/2)
    Wn = band/(Fs/2)
    
    #filterOrder = 8
    sos = spsignal.butter(filterOrder/2, Wn, 'bandpass', output = 'sos')
    
    data1_filt = spsignal.sosfiltfilt(sos, data1, axis=-1)
    
    data2_filt = spsignal.sosfiltfilt(sos, data2, axis=-1)
    
    data1_filt_hil = spsignal.hilbert(data1_filt, axis=-1)
    data2_filt_hil = spsignal.hilbert(data2_filt, axis=-1)
    
    phase_diff = (np.angle(data1_filt_hil) - np.angle(data2_filt_hil) + np.pi)%(2*np.pi) - np.pi # phase1 - phase2
    
    return data1_filt, data2_filt, phase_diff


#%%
"""
def PAC_2_band(raw_signal, phase_f, amp_f, Fs = 10000, eliminate_phase_delay=True, return_amp_phase_dist=False):
    '''
    do Phase-amplitude coupling (PAC) analysis, the type of filter used in PAC is Butterworth
    input:
    raw_signal: 1-d array, input raw signal
    phase_f: 1*2 array, define the cut-off frequency for phase-frequency band
    amp_f: 1*2 array, define the cut-off frequency for amplitude-frequency band
    Fs: sampling frequency
    eliminate_phase_shift: if using 'sosfiltfilt' to eliminate phase distortion(or time shift) caused by filter
    IIR filter usually causes phase distortion between original signal and filtered signal, use forward-backward method-'sosfiltfilt' to aviod it
    return_amp_phase_dist: if return the distribution of amplitude versus phase
    output:
    PAC_index: the higher the PAC index, the stronger the phase-amplitude coupling between phase-frequency and amplitude-frequency
    norm_mean_amp_per_phase: distribution of amplitude versus phase
    '''
    
    band_phase = np.asarray(phase_f) #np.array([phase_f_bin[j], phase_f_bin[j+1]])
    band_amp = np.asarray(amp_f) #np.array([amp_f_bin[i], amp_f_bin[i+1]])
    Wn_phase = band_phase/(Fs/2)
    Wn_amp = band_amp/(Fs/2)
    #pdb.set_trace()
    sos_phase = spsignal.butter(4, Wn_phase, 'bandpass', output='sos')
    sos_amp = spsignal.butter(4, Wn_amp, 'bandpass', output='sos')
    
    if eliminate_phase_delay:
        sig_phase = spsignal.sosfiltfilt(sos_phase, raw_signal)
        sig_amp = spsignal.sosfiltfilt(sos_amp, raw_signal)        
    else:
        sig_phase = spsignal.sosfilt(sos_phase, raw_signal)
        sig_amp = spsignal.sosfilt(sos_amp, raw_signal)
    #pdb.set_trace()            
    sig_phase_a = spsignal.hilbert(sig_phase)
    phase_sig_phase = np.angle(sig_phase_a)
    
    sig_amp_a = spsignal.hilbert(sig_amp)
    amp_sig_amp = np.abs(sig_amp_a)
    
    phase_bin = np.linspace(-np.pi, np.pi, 21)
    mean_amp_per_phase = np.zeros(len(phase_bin)-1)
    #pdb.set_trace()
    for k in range(len(phase_bin)-1):
        mean_amp_per_phase[k] = np.mean(amp_sig_amp[(phase_sig_phase >= phase_bin[k]) & (phase_sig_phase < phase_bin[k+1])])
    
    norm_mean_amp_per_phase =  mean_amp_per_phase/np.sum(mean_amp_per_phase)
    
    entropy_norm_mean_amp_per_phase = np.sum(norm_mean_amp_per_phase*np.log2(norm_mean_amp_per_phase))*(-1)
    entropy_uniform = np.log2(len(norm_mean_amp_per_phase))
    #pdb.set_trace()
    PAC_index = 1 - (entropy_norm_mean_amp_per_phase/entropy_uniform)
    
    if return_amp_phase_dist: return PAC_index, norm_mean_amp_per_phase
    else: return PAC_index 


def PAC_multi_band(raw_signal, phase_f_bin, amp_f_bin, Fs = 10000):
    '''
    do Phase-amplitude coupling analysis
    input:
    raw_signal: 1-d array, input raw signal
    phase_f_bin: 1-d array, bins for phase-frequency
    amp_f_bin: 1-d array, bins for amplitude-frequency
    Fs: sampling frequency
    output:
    PAC_mat: 2-d array, size: len(amp_f_bin) * len(phase_f_bin), PAC index for different phase-frequency and amplitude-frequency
    the higher the PAC index, the stronger the phase-amplitude coupling between phase-frequency and amplitude-frequency
    '''
    PAC_mat = np.zeros([len(amp_f_bin)-1, len(phase_f_bin)-1])

    for i in range(len(amp_f_bin)-1):
        for j in range(len(phase_f_bin)-1):
            ##amp_Wn = amp_f_bin[i], amp_f_bin[i+1]
            #phase_f_bin[j], phase_f_bin[j+1]
            #pdb.set_trace()
            phase_f = np.array([phase_f_bin[j], phase_f_bin[j+1]])
            amp_f = np.array([amp_f_bin[i], amp_f_bin[i+1]])
            PAC_index = PAC_2_band(raw_signal, phase_f, amp_f, Fs = Fs, return_amp_phase_dist=False)
            PAC_mat[i,j] = PAC_index
            
#            band_phase = np.array([phase_f_bin[j], phase_f_bin[j+1]])
#            band_amp = np.array([amp_f_bin[i], amp_f_bin[i+1]])
#            Wn_phase = band_phase/(Fs/2)
#            Wn_amp = band_amp/(Fs/2)
#            #pdb.set_trace()
#            sos_phase = spsignal.butter(4, Wn_phase, 'bandpass', output='sos')
#            sos_amp = spsignal.butter(4, Wn_amp, 'bandpass', output='sos')
#            
#            sig_phase = spsignal.sosfilt(sos_phase, raw_signal)
#            sig_amp = spsignal.sosfilt(sos_amp, raw_signal)
#            #pdb.set_trace()            
#            sig_phase_a = spsignal.hilbert(sig_phase)
#            phase_sig_phase = np.angle(sig_phase_a)
#            
#            sig_amp_a = spsignal.hilbert(sig_amp)
#            amp_sig_amp = np.abs(sig_amp_a)
#            
#            phase_bin = np.linspace(-np.pi, np.pi, 21)
#            amp_mean_perbin = np.zeros(len(phase_bin)-1)
#            #pdb.set_trace()
#            for k in range(len(phase_bin)-1):
#                amp_mean_perbin[k] = np.mean(amp_sig_amp[(phase_sig_phase >= phase_bin[k]) & (phase_sig_phase < phase_bin[k+1])])
#            
#            norm_amp_mean_perbin =  amp_mean_perbin/np.sum(amp_mean_perbin)
#            
#            entropy_amp_mean_perbin = np.sum(norm_amp_mean_perbin*np.log2(norm_amp_mean_perbin))*(-1)
#            entropy_uniform = np.log2(len(norm_amp_mean_perbin))
#            #pdb.set_trace()
#            PAC_index = 1 - (entropy_amp_mean_perbin/entropy_uniform)
#            PAC_mat[i,j] = PAC_index

    #return phase_sig_phase, amp_sig_amp, amp_mean_perbin, entropy_amp_mean_perbin, PAC_index
    return PAC_mat
#%%
"""




    
