# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:56:52 2020

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
import levy
#%%
sys_argv = int(sys.argv[1])
loop_num = sys_argv

#i = 2; j = 0; k = 2
#data_num = i*2*9+j*9+k
#fileadr = '/import/headnode1/shni2598/brian2/data/onearea_chg_adapt_simp_2/'
with open('data%s.file'%(loop_num), 'rb') as file:
    spke1 = load_data_dict.data_onegroup(pickle.load(file))

spke1.t = spke1.t*0.1*ms
#%%
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spke1, starttime=0*ms, endtime=10*second, binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)

jump_size1 = jump_size1.reshape(-1)
jump_size1 = np.concatenate((jump_size1, jump_size1*(-1)), 0)
#%%
fit_results = levy.fit_levy(jump_size1)
fit_results_list = list(fit_results[0].get('0'))
fit_results_list = [str(num)+',' for num in fit_results_list]
fit_results_list.append(str(fit_results[1])+',\n')
fit_results_list.insert(0, str(spke1.param['ie_ratio'])+',')
fit_results_list.insert(0, str(loop_num)+',')

with open('fitjumping.txt','a') as file:
    file.writelines(fit_results_list)
