# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:16:08 2020

@author: nishe
"""

import load_data_dict
import brian2.numpy_ as np
from brian2.only import *
import post_analysis as psa
import connection as cn
import pickle
import matplotlib as mpl
mpl.use('Agg')
import sys
import matplotlib.pyplot as plt
from scipy.fftpack import fft
#%%
lattice_ext = cn.coordination.makelattice(63, 62, [0, 0])
#%%
chg_adapt_loca = [0, 0]
chg_adapt_range = 10
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(lattice_ext, chg_adapt_loca, chg_adapt_range, 62)

#%%
ie_ratio = 3.375*np.arange(0.9, 1.21, 0.03)
tau_s_di = [3., 6.] # ms
new_delta_gk = np.arange(0, 8.1, 1) # nS
#%%
#i = 10; j = 1; k = 8
sys_argv = int(sys.argv[1])
loop_num = -1
for i in range(len(ie_ratio)):
    for j in range(len(tau_s_di)):
        for k in range(len(new_delta_gk)):
            loop_num += 1
            if loop_num == sys_argv:
                print(i,j,k)
                break
        else: continue
        break
    else:continue
    break

#i = 2; j = 0; k = 2
#data_num = i*2*9+j*9+k
#%%
loop_num = 1233
with open('data%s'%(loop_num), 'rb') as file:
    spke1 = load_data_dict.data_onegroup(pickle.load(file))

spke1.t = spke1.t*0.1*ms

#%%
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spke1, starttime=0*ms, endtime=6000*ms, binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)
#spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)
#anititle='change adaptation in region near centre to %.1fnS at 1000 ms\ntau_s_di: %.1fms; ie_ratio: %.3f'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i])
#ani1 = psa.show_pattern(spkrate1, spkrate2=0, area_num = 1, frames = 5500, bottom_up=1, top_down=1, stimu_onset = -1, start_time = 0, anititle=anititle)
#ani1.save('chg_adapt_1_%.1f_%.1f_%.3f.mp4'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i]))
#%%

rate_chgadapt_neu = spkrate1.reshape(3969,-1)[chg_adapt_neuron]
poprate_rate_chgadapt_neu = rate_chgadapt_neu.sum(0)/len(chg_adapt_neuron)/0.01
#%%
plt.figure(figsize=[9,6])
plt.plot(np.arange(len(poprate_rate_chgadapt_neu)),poprate_rate_chgadapt_neu)
plt.plot([1000,1000],[0,140],'r--', label='time of adaptation change')
plt.xlabel('time(ms)')
plt.ylabel('rate(Hz)')
plt.title('firing rate of neurons with decreased adaptation\ndelta_gk: %.1f\ntau_s_di: %.1fms; ie_ratio: %.3f'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i]))
plt.legend()
#plt.savefig('rate_chg_adapt_1_%.1f_%.1f_%.3f.png'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i]))
#%%
magf = abs(fft(poprate_rate_chgadapt_neu[1000:]-poprate_rate_chgadapt_neu[1000:].mean()))
plt.figure(figsize=[9,6])
plt.plot((np.arange(len(magf))/(len(magf))*1000)[:250], magf[:250])
plt.xlabel('freqeuncy(Hz)')
plt.title('Fourier Transform of rate of neurons after adaptation change\ndelta_gk: %.1f\ntau_s_di: %.1fms; ie_ratio: %.3f'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i]))
#plt.savefig('fftrate_chg_adapt_1_%.1f_%.1f_%.3f.png'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i]))
#%%
#plt.figure()
#plt.loglog((np.arange(len(magf))/(len(magf))*1000)[:2400], magf[:2400],'*')
#%%
spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)
anititle='change adaptation in region near centre to %.1fnS at 1000 ms\ntau_s_di: %.1fms; ie_ratio: %.3f'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i])
ani1 = psa.show_pattern(spkrate1, spkrate2=0, area_num = 1, frames = 5500, bottom_up=1, top_down=1, stimu_onset = -1, start_time = 0, anititle=anititle)
#ani1.save('chg_adapt_1_%.1f_%.1f_%.3f.mp4'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i]))
#%%
spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)
anititle='change adaptation in region near centre to %.1fnS at 1000 ms\ndelta_gk: %.1fnS; tau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio'])
ani1 = psa.show_pattern(spkrate1, spkrate2=0, area_num = 1, frames = 5500, bottom_up=1, top_down=1, stimu_onset = -1, start_time = 0, anititle=anititle)
#ani1.save('chg_adapt_%d_%.1f_%.1f_%.1f_%.1f_%.3f_%d.png'%savename)




