# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:47:31 2020

@author: nishe
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 21:29:12 2020

@author: nishe
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mydata
import frequency_analysis#.myfft
import firing_rate_analysis
import connection as cn
import sys

#%%
datapath = ''
#datapath = '/import/headnode1/shni2598/brian2/data/attention/compare_AD_YG/chg_adapt/'
loop_num = int(sys.argv[1])

data = mydata.mydata()
data.load(datapath+'data%d.file'%(loop_num))
#%%
lattice_ext = cn.coordination.makelattice(63,62,[0,0])
chg_adapt_loca = [0, 0]
chg_adapt_range = 6
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(lattice_ext, chg_adapt_loca, chg_adapt_range, 62)
#%%
data.a1.ge.get_spike_rate(start_time=0, end_time=30000,\
                           sample_interval = 1, n_neuron = 3969, window = 5, dt = 0.1, reshape_indiv_rate = False)

MUA = data.a1.ge.spk_rate.spk_rate[chg_adapt_neuron].sum(0)

#%%
adapt_chg_t = int(np.round(10*1000/1))
fig, [ax1,ax2] = plt.subplots(2,1, figsize=[15,7]) 
ax1.plot(np.arange(len(MUA[:adapt_chg_t]))*1,  MUA[:adapt_chg_t])
ax2.plot(np.arange(len(MUA[adapt_chg_t:]))*1,  MUA[adapt_chg_t:])
ax1.set_title('no adaptation change')
ax2.set_title('adaptation change')
fig.savefig('MUA_%.4f_%d.png'%(data.a1.param.ie_ratio,loop_num))
#%%
discard_t = int(np.round(2000/1))

coef, freq = frequency_analysis.myfft(MUA[:adapt_chg_t][discard_t:], 1000)
coef = coef/len(MUA[:adapt_chg_t][discard_t:])

fig, ax = plt.subplots(1,1, figsize=[8,7]) 
ax.plot(freq[1:1000], np.abs(coef[1:1000]), label = 'no adaptation change')

coef, freq = frequency_analysis.myfft(MUA[adapt_chg_t:][discard_t:], 1000)
coef = coef/len(MUA[adapt_chg_t:][discard_t:])

ax.plot(freq[1:2000], np.abs(coef[1:2000]), label = 'adaptation change')
ax.legend()
fig.savefig('MUA_fft_%.4f_%d.png'%(data.a1.param.ie_ratio,loop_num))

#%%
''' animation  '''

data.a1.ge.spk_rate.spk_rate = data.a1.ge.spk_rate.spk_rate.reshape(63,63,-1)

#%%
anititle = '''ie_ratio: %.4f, 
adaptation change at centre at: 10000 ms'''%(data.a1.param.ie_ratio) 
start_time = 9000; end_time = 12000; frames = end_time - start_time
ani = firing_rate_analysis.show_pattern(data.a1.ge.spk_rate.spk_rate[:,:,start_time:end_time], 
                                   frames=frames, start_time=start_time, anititle=anititle)

ani.save('pattern_%.4f_%d.mp4'%(data.a1.param.ie_ratio,loop_num))








