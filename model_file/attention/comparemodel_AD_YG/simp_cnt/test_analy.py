#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:14:50 2020

@author: shni2598
"""
import matplotlib as mpl
#mpl.use('Agg')
#import load_data_dict
import mydata
import brian2.numpy_ as np
from brian2.only import *
#import post_analysis as psa
import firing_rate_analysis as fa
import connection as cn
import pickle
import sys
import os
import matplotlib.pyplot as plt
import levy

import matplotlib.animation as animation

#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/chg_p_decay_2/'

#sys_argv = int(sys.argv[1])
loop_num = 564 #rep_ind*20 + ie_num
#%%
data = mydata.mydata()
data.load(datapath+'data%d.file'%loop_num)
#%%
e_lattice = cn.coordination.makelattice(63,62,[0,0])
chg_adapt_loca = [0, 0]
chg_adapt_range = 6
width = 62
chg_adapt_neuron = cn.findnearbyneuron.findnearbyneuron(e_lattice, chg_adapt_loca, chg_adapt_range, width)

#%%
data.a1.ge.get_spike_rate(start_time=1e3, end_time=10e3,\
                           sample_interval = 1, n_neuron = 3969, window = 30, dt = 0.1, reshape_indiv_rate = True)

data.a1.ge.get_centre_mass(slide_interval=1, jump_interval=15)
#%%
start_time=13e3; end_time=14e3
data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time,\
                           sample_interval = 1, n_neuron = 3969, window = 15, dt = 0.1, reshape_indiv_rate = True)
data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time,\
                           sample_interval = 1, n_neuron = 1024, window = 15, dt = 0.1, reshape_indiv_rate = True)

#%%
frames = int(end_time-start_time)
#ani = fa.show_pattern(data.a1.ge.spk_rate.spk_rate, frames=frames-50, start_time=start_time)

ani = fa.show_pattern(data.a1.ge.spk_rate.spk_rate, data.a1.gi.spk_rate.spk_rate, frames=frames-50, start_time=start_time)
#%%
fit_py = levy.fit_levy(data.a1.ge.centre_mass.jump_size.reshape(-1))
#%%
data.a1.ge.get_MSD(start_time=1000, end_time=10000, jump_interval=np.arange(1,1000,2), fit_stableDist=None)
#%%
data.a1.ge.get_MSD(start_time=1000, end_time=10000,  window = 30, jump_interval=np.array([15,30]), fit_stableDist='Matlab')
data.a1.ge.get_MSD(start_time=1000, end_time=10000,  window = 30, jump_interval=np.array([15,30]), fit_stableDist='pylevy')
#%%
data.a1.ge.get_spike_rate(start_time=1e3, end_time=20e3,\
                           sample_interval = 1, n_neuron = 3969, window = 15, dt = 0.1, reshape_indiv_rate = True)
#%%
mua = data.a1.ge.spk_rate.spk_rate.reshape(3969, -1)[chg_adapt_neuron]
mua = mua.sum(0)
plt.figure()
plt.plot(mua)










#%%
import matlab.engine

eng = matlab.engine.start_matlab('-nodisplay') # '-nodisplay'

jumpsize_mat = matlab.double(list(data.a1.ge.centre_mass.jump_size.reshape(-1)), size=[data.a1.ge.centre_mass.jump_size.size,1])
#distparam[i] = my_fitstable.fitstable(inputIn)

#jumpsize_mat = jump_size1.reshape(-1)
#plt.figure()
#plt.hist(jumpsize_mat)
#jumpsize_mat = matlab.double(list(jumpsize_mat))
#jumpsize_mat.reshape([2*len(jump_size1),1])

#eng.fitdist(jumpsize_mat,'stable',nargout=0)
eng.workspace['result_fit'] = eng.fitdist(jumpsize_mat,'stable')#,nargout=0)

#eng.eval("result_fit=fitdist(jumpsize_mat,'stable')", nargout=0)
paramci = eng.eval('result_fit.paramci')
fitparam = eng.eval('[result_fit.alpha result_fit.beta result_fit.gam result_fit.delta]')
fitparam = np.concatenate((np.array(fitparam),np.array(paramci)),0)      
#%%
import fitStable_mat
#%%
eng = fitStable_mat.start_Mat_Eng()
fit_mat, eng = fitStable_mat.fitStable(data.a1.ge.centre_mass.jump_size.reshape(-1), eng)
#%%
fig,ax = plt.subplots(1,1)
ax = firing_rate_analysis.plot_traj(ax, data.a1.ge.centre_mass.centre_ind[:1000])
#%%
pre_cent = ax, data.a1.ge.centre_mass.centre_ind
#%%
fig,ax = plt.subplots(1,1)
ax = firing_rate_analysis.plot_traj(ax, pre_cent[1][:1000])
ax.set_xlim([-0.5,62.5])
ax.set_ylim([-0.5,62.5])







#%%



from matplotlib.animation import FuncAnimation
#%%
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 512),
                    init_func=init, blit=True, interval=2)
plt.show()
#%%
spkrate1=data.a1.ge.spk_rate.spk_rate; 
frames = int(end_time-start_time)
start_time = start_time; anititle='animation'
#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
fig, ax1= plt.subplots(1,1, figsize = (6,6))
#divider = make_axes_locatable(ax1)
#cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
#fig.colorbar(img1, cax=cax1)

cmap_spk=plt.cm.get_cmap('Blues', 7) # viridis Blues
cmap_c = np.array([1.,0.,0.,1.])
cmap_stimulus = np.array([88/255,150/255.,0.,1.])
cmap = np.vstack((cmap_stimulus,cmap_c,cmap_spk(range(7))))
cmap = mpl.colors.ListedColormap(cmap)
#cmap.set_under('red')
bounds = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.03]) 

cb = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                norm=norm,
                                boundaries=bounds,
                                ticks=np.array([-2,-1, 0, 1, 2, 3, 4, 5, 6])+0.5,
                                spacing='proportional',
                                orientation='horizontal') #horizontal vertical
cb.ax.set_xticklabels(['stimulus','ctr', 0, 1, 2, 3, 4, 5, 6])
cb.set_label('number of spikes')

#titleaxes = fig.add_axes([0.3, 0.75, 0.4, 0.05])
#titleaxes = divider.append_axes("top", size="5%", pad=0.01)
titleaxes = fig.add_axes([0.3, 0.85, 0.4, 0.05])
titleaxes.axis('off')
title = titleaxes.text(0.5,0.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=titleaxes.transAxes, ha="center")
time_title = np.arange(spkrate1.shape[2]) + start_time

value1=ax1.matshow(spkrate1[:,:,0], cmap=cb.cmap, norm=cb.norm)
#value2=ax2.matshow(spk2[:,:,:], cmap=cb.cmap)

def init():
    value1.set_array(spkrate1[:,:,0])
    title.set_text(u"time: {} ms".format(time_title[0]))
    #title.set_text(u"time: {} ms".format(''))
    return [value1, title,]

def updatev(i):
    title.set_text(u"time: {} ms".format(time_title[i]))
    #title.set_text(u"time: {} ms".format(''))
    value1.set_array(spkrate1[:,:,i])
    #value2.set_array(spk2[:,:,i])
    return [value1, title,] #, value2

#value1.set_clim(vmin=0, vmax=6)
ax1.axis('off')
#if stimu_onset >= 0: fig.suptitle('onset of stimulus:%dms'%(stimu_onset))
#fig.suptitle(anititle)
ani=animation.FuncAnimation(fig, updatev, init_func=init, frames=spkrate1.shape[-1], interval=10, repeat_delay = 2000, blit=True)    # frames=spk1.shape[2]
#ani=animation.FuncAnimation(fig, updatev, init_func=init, frames=frames, interval=10)#, blit=True)    # frames=spk1.shape[2]

#%%
#%%
from brian2.numpy_ import sin, cos
#import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()
#%%
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg


def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2] - state[0]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
               M2*G*sin(state[2])*cos(del_) +
               M2*L2*state[3]*state[3]*sin(del_) -
               (M1 + M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[0])*cos(del_) -
               (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
               (M1 + M2)*G*sin(state[2]))/den2

    return dydx

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange(0.0, 20, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)

x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])

x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'

#time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
time_text_ax = fig.add_axes([0.4,0.9,0.2,0.1])
time_text = time_text_ax.text(0.05, 0.05, '')#, transform=time_text_ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=25, blit=True, init_func=init)

# ani.save('double_pendulum.mp4', fps=15)
plt.show()
