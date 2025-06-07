# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:56:08 2020

@author: nishe
"""
'''
Change the synapse model in Yifan's model to linear differential equation
'''
#%%
from brian2.only import *
import brian2.numpy_ as np
import matplotlib.pyplot as plt
import pre_process
import poisson_stimuli as psti
import connection as cn
import post_analysis as psa
from scipy.fftpack import fft
import levy
#%%
scale_ee = 1.4; scale_ei = 1.4; scale_ii =1.4; scale_ie_ratio = 1

#delta_gk
#new_delta_gk
#tau_s_de
#tau_s_di

ijwd = pre_process.get_ijwd(Ni=1024)
ijwd.w_ee_dist = 'lognormal'
ijwd.hybrid = 0.
ijwd.cn_scale_weight = 1
ijwd.cn_scale_wire = 1
ijwd.iter_num=1

#scale_ee = 1.4; scale_ei = 1.4
ie_ratio=3.375 * scale_ie_ratio#* scale_ie/scale_ee
ijwd.ie_ratio = ie_ratio# 3.375 #3.375 #* scale_ie_ratio[i]
ijwd.mean_J_ee = 4*10**-3 * scale_ee#* scale_ee#* 1.4 # usiemens
ijwd.sigma_J_ee = 1.9*10**-3 * scale_ee#* scale_ee#* 1.4# usiemens

ijwd.generate_ijw()
ijwd.generate_d_rand()

ijwd.w_ei = 5*10**(-3) * scale_ei * 1.05#* scale_ei * 1.05#usiemens
ijwd.w_ii = 25*10**(-3) * scale_ii #* 1.4  #usiemens

#%%
start_scope()

neuron_linear_E = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + (-g_k)*(v - v_k) + I_i + I_e + I_extnl) : volt (unless refractory)
dg_k/dt = -g_k/tau_k : siemens
tau_k : second
delta_gk : siemens

I_i = (-g_I)*(v - v_rev_I) : amp
I_e = (-g_E-g_E_inter-g_E_extnl)*(v - v_rev_E) : amp

dg_I/dt = -g_I/tau_s_di + x_I/tau_s_di : siemens
dx_I/dt = -x_I/tau_s_ri : siemens
dg_E/dt = -g_E/tau_s_de + x_E/tau_s_de : siemens
dx_E/dt = -x_E/tau_s_re : siemens

g_E_inter : siemens

I_extnl : amp
g_E_extnl: siemens
'''
neuron_linear_I = '''
dv/dt = (1/C)*(-g_l*(v - v_l) + I_i + I_e + I_extnl) : volt (unless refractory)

I_i = (-g_I)*(v - v_rev_I) : amp
I_e = (-g_E-g_E_inter-g_E_extnl)*(v - v_rev_E) : amp

dg_I/dt = -g_I/tau_s_di + x_I/tau_s_di : siemens
dx_I/dt = -x_I/tau_s_ri : siemens
dg_E/dt = -g_E/tau_s_de + x_E/tau_s_de : siemens
dx_E/dt = -x_E/tau_s_re : siemens

g_E_inter : siemens

I_extnl : amp
g_E_extnl: siemens
'''
synapse='''
w : siemens'''
#%%
tau_s_de = 5*ms
tau_s_di = 3*ms # Yuxi's parameter
tau_s_re = 1*ms
tau_s_ri = 1*ms
#%%
group_e =NeuronGroup(ijwd.Ne, model=neuron_linear_E,
                 threshold='v>v_threshold', method='euler',
                 reset='''v = v_reset
                          g_k += delta_gk''', refractory='(t-lastspike)<t_ref')

group_i =NeuronGroup(ijwd.Ni, model=neuron_linear_I,
                     threshold='v>v_threshold', method='euler',
                     reset='v = v_reset', refractory='(t-lastspike)<t_ref')
syn_ee = Synapses(group_e, group_e, model=synapse, method='euler',
                  on_pre='''x_E_post += w''')
syn_ei = Synapses(group_e, group_i, model=synapse, method='euler',
                  on_pre='''x_E_post += w''')
syn_ie = Synapses(group_i, group_e, model=synapse, method='euler',
                  on_pre='''x_I_post += w''')
syn_ii = Synapses(group_i, group_i, model=synapse, method='euler',
                  on_pre='''x_I_post += w''')

delta_x_extnl = 2*nS*tau_s_de/tau_s_re
group_input = PoissonGroup(ijwd.Ne, psti.input_spkrate(maxrate = 600, sig=6, position=[[-31.5, -31.5],[0, 0]])*Hz)
syn_extnl_e = Synapses(group_input, group_e, method='euler', on_pre='x_E_post += delta_x_extnl')
syn_extnl_e.connect('i==j')

syn_ee.connect(i=ijwd.i_ee, j=ijwd.j_ee)
syn_ei.connect(i=ijwd.i_ei, j=ijwd.j_ei)
syn_ie.connect(i=ijwd.i_ie, j=ijwd.j_ie)
syn_ii.connect(i=ijwd.i_ii, j=ijwd.j_ii)

syn_ee.w[:] = ijwd.w_ee*usiemens*(5*ms)/tau_s_re
syn_ei.w[:] = ijwd.w_ei*usiemens*(5*ms)/tau_s_re #5*nS
syn_ie.w[:] = ijwd.w_ie*usiemens*(3*ms)/tau_s_ri #25*nS
syn_ii.w[:] = ijwd.w_ii*usiemens*(3*ms)/tau_s_ri
#%%
syn_ee.delay = ijwd.d_ee*ms
#syn_ee = set_delay(syn_ee, ijwd.d_ee)
syn_ie.delay = ijwd.d_ie*ms
syn_ei.delay = ijwd.d_ei*ms
syn_ii.delay = ijwd.d_ii*ms
#%%
group_e.v[:] = np.random.random(ijwd.Ne)*35*mV-85*mV
group_i.v[:] = np.random.random(ijwd.Ni)*35*mV-85*mV
group_e.delta_gk = 10*nS
group_e.tau_k = 80*ms

group_e.I_extnl = 0.51*nA #0.51*nA
group_i.I_extnl = 0.60*nA #0.60*nA
#%%
C = 0.25*nF # capacitance
g_l = 16.7*nS # leak capacitance
v_l = -70*mV # leak voltage
v_threshold = -50*mV
v_reset = -60*mV
v_rev_I = -80*mV
v_rev_E = 0*mV
v_k = -85*mV
#tau_k = 80*ms# 80*ms
#delta_gk = 10*nS #10*nS
t_ref = 4*ms # refractory period
#tau_x_re = 1*ms
#u_base = 0.2
#tau_f = 1500*ms; tau_d = 200*ms
#tau_s_re = 1*ms; tau_s_de = 5*ms
#delta_gk = 0*nS#10*nS #10*nS Yuxi's parameter
#%%
chg_adapt_loca = [0, 0]
chg_adapt_range = 7.
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.lattice_ext, chg_adapt_loca, chg_adapt_range, ijwd.width)

#%%
spk_e1 = SpikeMonitor(group_e, record = True)
spk_i1 = SpikeMonitor(group_i, record = True)
#%%
network = Network(collect())
network.store('state1')
#%%
ijwd.change_ie(3.375)
syn_ie.w[:] = ijwd.w_ie*usiemens*(3*ms)/tau_s_ri #25*nS
network.store('state1')
#%%
group_input.active = 0#False
network.run(1000*ms,report='text')

#group_input.active = 1#False
group_e.delta_gk[chg_adapt_neuron] = 2*nS; group_e.tau_k[chg_adapt_neuron] = 30*ms
network.run(4000*ms,report='text')

#%%
start_time = 0*ms; end_time=5000*ms
spkratei1 = psa.get_spike_rate(spk_i1, start_time=start_time, end_time=end_time, indiv_rate = True, popu_rate = False, \
                   sample_interval = 1*ms, n_neuron = 1024, window = 10*ms, dt = 0.1*ms)

spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spk_e1, starttime=start_time, endtime=end_time, binforrate=30*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)
spkrate1 = psa.overlap_centreandspike(centre_ind1, spkrate1, show_trajectory = False)
#anititle='change adaptation in region near centre to %.1fnS at 1000 ms\ntau_s_di: %.1fms; ie_ratio: %.3f'%(new_delta_gk, tau_s_di/ms, ie_ratio)
anititle=''
ani2 = psa.show_pattern(spkrate1=spkrate1, spkrate2=spkratei1, area_num = 2, frames = 3800, bottom_up=1, top_down=1, stimu_onset = -1, start_time = 0, anititle=anititle)
#%%
jump_size1 = jump_size1.reshape(-1)
jump_size1 = np.concatenate((jump_size1, jump_size1*(-1)), 0)
#%%
fit_br = levy.fit_levy(jump_size1)
fit_br#fit_br[0].get('0')
#%%
plt.figure()
stat = plt.hist2d(centre_ind1[1000:,1], centre_ind1[1000:,0], bins=np.linspace(0,62,10))
#plt.matshow(stat[0])
plt.colorbar()
#%%
plt.matshow(stat[0]/len(centre_ind1[1000:]), extent=[0,62,62,0])
plt.title('residence time of pattern at\n different locations in the network (%)')
ax = plt.gca()
ax.xaxis.set_ticks_position('bottom')
#ax.tick_params(labeltop=False)
plt.colorbar()
#%%
rate_chgadapt_neu = spkrate1.reshape(3969,-1)[chg_adapt_neuron]
poprate_rate_chgadapt_neu = rate_chgadapt_neu.sum(0)/len(chg_adapt_neuron)/0.01

plt.figure(figsize=[12,6])
plt.plot(np.arange(len(poprate_rate_chgadapt_neu)),poprate_rate_chgadapt_neu)
plt.plot([1000,1000],[0,140],'r--', label='time of adaptation change')
plt.xlabel('time(ms)')
plt.ylabel('rate(Hz)')
#title = ''''adpt_range:%d,new_tau_k:%.0f,new_delta_gk:%.0f,delta_gk:%.0f
#tau_s_di:%.1f,tau_s_de:%.1f,ie_ratio:%.3f'''%savename
plt.title(title)#('firing rate of neurons with decreased adaptation\nnew_delta_gk:%.1f;delta_gk:%.1f\ntau_s_di: %.1fms; ie_ratio: %.3f'%(spke1.param['new_delta_gk'], spke1.param['delta_gk'], spke1.param['tau_s_di'], spke1.param['ie_ratio']))
plt.legend()
#%%
fig, ax = plt.subplots(1)
psa.plot_traj(ax, centre_ind1)
ax.set_xlim([-0.5,62.5])
ax.set_ylim([-0.5,62.5])
#%%
network.restore('state1')
#%%
ani1 = psa.show_pattern(spkrate1, spkrate2=0, area_num = 1, frames = 1800, bottom_up=1, top_down=1, stimu_onset = -1, start_time = 0, anititle=anititle)
#%%
plt.figure()
plt.plot(spk_e1.t/ms, spk_e1.i,'.')

#%%
mua_loca = [0, 0]
mua_range = 5
mua_neuron = cn.findnearbyneuron.findnearbyneuron(ijwd.lattice_ext, mua_loca, mua_range, ijwd.width)

mua = spkrate1.reshape(3969,-1)[mua_neuron]
poprate_rate_mua = mua.sum(0)/len(mua)/0.01
plt.figure(figsize=[9,6])
plt.plot(np.arange(len(poprate_rate_mua)),poprate_rate_mua)
plt.plot([1000,1000],[0,140],'r--', label='time of adaptation change')
plt.xlabel('time(ms)')
plt.ylabel('rate(Hz)')
plt.title('firing rate of neurons with decreased adaptation\ndelta_gk: %.1f\ntau_s_di: %.1fms; ie_ratio: %.3f'%(new_delta_gk[k], tau_s_di[j], ie_ratio[i]))
plt.legend()
#%%
spkrate1, centre_ind1, jump_size1, jump_dist1 = psa.get_rate_centre_jumpdist(spk_e1, starttime=2000*ms, endtime=8000*ms, binforrate=10*ms, sample_interval=1*ms, slide_interval = 1, jump_interval = 15, dt=0.1*ms, show_trajectory=False)
#%%
#plt.figure()
#plt.hist2d(centre_ind1[:,1], centre_ind1[:,0], bins=10)
plt.figure()
stat = plt.hist2d(centre_ind1[:,1], centre_ind1[:,0], bins=np.linspace(0,62,10))
#plt.matshow(stat[0])
plt.colorbar()
#%%
plt.matshow(stat[0]/len(centre_ind1))
plt.colorbar()
#%%
plt.figure()
plt.plot(centre_ind1[:,1].T, centre_ind1[:,0].T, '.')
plt.xlim([0,62])
plt.ylim([0,62])

#%%
fftcoef = fft(poprate_rate_mua[:])
magcoef = abs(fftcoef)
plt.figure()
plt.plot(np.arange(len(magcoef))[:int(len(magcoef)/2)]/len(magcoef)*1000,magcoef[:int(0.5*len(magcoef))])
plt.yscale('log')
plt.xscale('log')
#%%
fftcoef = fft(centre_ind1[:,1])
magcoef = abs(fftcoef)
plt.figure()
plt.plot(np.arange(len(magcoef))/len(magcoef)*1000,magcoef)
#%%
plt.figure()
plt.plot(centre_ind1[:,1])
#%%
import scipy.io as sio
#%%
sio.savemat('jumpsize_br.mat', {'jumpsize': jump_size1})

#%%
network.restore('state1')







