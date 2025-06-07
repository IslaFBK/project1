#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:45:43 2020

@author: shni2598
"""
import brian2.numpy_ as np
import matplotlib.pyplot as plt
#%%
e_lattice = cn.coordination.makelattice(64, 64, [0,0])
i_lattice = cn.coordination.makelattice(64, 64, [0,0])
#%%
dist = cn.coordination.lattice_dist(e_lattice, 64, [-0.5,0])
#%%
dist = cn.coordination.lattice_dist(i_lattice, 64, [0,0])
#%%
tau_d = 3.5
p_peak = 1
print(p_peak*(np.sum(np.exp(-dist/tau_d)))/2)

minii=93; maxii=177

#%%
tau_d = 10#5.76#10*0.82*1.15
p_peak = 0.15
print(p_peak*(np.sum(np.exp(-dist/tau_d)))/3)

minii=93; maxii=177
#%%
tau_d = 20#20*0.86*1.15#20*0.86*1.15
p_peak = 0.57*0.8#*1.05#0.84
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
tau_d = 8
p_peak = 0.85#0.84
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
tau_d = np.arange(5,25)
num = np.zeros(tau_d.shape[0])
p_peak = 0.85#0.84
for ii in range(tau_d.shape[0]):
    num[ii] = p_peak*(np.sum(np.exp(-dist/tau_d[ii])))
#%%
plt.figure()
plt.plot(tau_d, num)
#%%
ee = 220; tau = 7#8*0.85  ; 
peak = 0.85*0.88 #0.85 * 0.85 
ei = 395; tau = 9.02#10*0.82*1.1 
peak = 1*0.82*1.05

ie = 150; tau = 20#20*0.86*1.15  ; 
peak = 0.57*0.8

ii = 260; tau = 20#20*0.86*1.15  ; 
peak = 1*0.78



#%%
tau_d = 8
p_peak = 0.85*0.5#0.84
print(p_peak*(np.sum(np.exp(-dist/tau_d))))
#%%
ee=317
ei=542
ie=200
ii=335

ee=182.4377
#%%
ee = 195
tau_d = 8*0.85
p_peak = 0.7
#%%
ei = 320
tau_d = 10*0.82
p_peak = 0.82
#%%
ie = 129
tau_d = 20*0.82
p_peak = 0.58*0.82
#%%
ii = 221
tau_d = 20*0.82
p_peak = 1*0.82
#%%
pk1 = 1; pk2 = 1*0.82
tau1 = 20; tau2 = 20*0.82
xx = np.linspace(0,31,100)
yy1 = pk1*np.exp(-xx/tau1)
yy2 = pk2*np.exp(-xx/(tau2))

fig, ax = plt.subplots(1,1)
ax.plot(xx,yy1, xx, yy2)
#%%
plt.figure()
tau_d = 8*0.85; p_peak = 0.7
yy = p_peak*np.exp(-xx/tau_d)
plt.plot(xx,yy, 'b', label='ee_n')
tau_d = 8; p_peak = 0.7/0.85
yy = p_peak*np.exp(-xx/tau_d)
plt.plot(xx,yy,'b--', label='ee')

tau_d = 10*0.82; p_peak = 0.82
yy = p_peak*np.exp(-xx/tau_d)
plt.plot(xx,yy, 'y', label='ei_n')
tau_d = 10; p_peak = 1
yy = p_peak*np.exp(-xx/tau_d)
plt.plot(xx,yy, 'y--', label='ei')

tau_d = 20*0.82; p_peak = 0.58*0.82
yy = p_peak*np.exp(-xx/tau_d)
plt.plot(xx,yy, 'g',label='ie_n')
tau_d = 20; p_peak = 0.58
yy = p_peak*np.exp(-xx/tau_d)
plt.plot(xx,yy, 'g--',label='ie')

tau_d = 20*0.82; p_peak = 1*0.82
yy = p_peak*np.exp(-xx/tau_d)
plt.plot(xx,yy, 'r',label='ii_n')
tau_d = 20; p_peak = 1
yy = p_peak*np.exp(-xx/tau_d)
plt.plot(xx,yy, 'r--',label='ii')

plt.legend()
#%%
fig, [ax1,ax2] = plt.subplots(1,2)
tau_d = 8*0.85; p_peak = 0.7
yy = p_peak*np.exp(-xx/tau_d)
ax1.plot(xx,yy, 'b', label='ee_n')

tau_d = 10*0.82; p_peak = 0.82
yy = p_peak*np.exp(-xx/tau_d)
ax1.plot(xx,yy, 'y', label='ei_n')

tau_d = 20*0.82; p_peak = 0.58*0.82
yy = p_peak*np.exp(-xx/tau_d)
ax1.plot(xx,yy, 'g',label='ie_n')

tau_d = 20*0.82; p_peak = 1*0.82
yy = p_peak*np.exp(-xx/tau_d)
ax1.plot(xx,yy, 'r',label='ii_n')
ax1.legend()
ax1.set_ylim([0,1.1])


tau_d = 8; p_peak = 0.82
yy = p_peak*np.exp(-xx/tau_d)
ax2.plot(xx,yy,'b--', label='ee')
tau_d = 10; p_peak = 1
yy = p_peak*np.exp(-xx/tau_d)
ax2.plot(xx,yy, 'y--', label='ei')
tau_d = 20; p_peak = 0.58
yy = p_peak*np.exp(-xx/tau_d)
ax2.plot(xx,yy, 'g--',label='ie')
tau_d = 20; p_peak = 1
yy = p_peak*np.exp(-xx/tau_d)
ax2.plot(xx,yy, 'r--',label='ii')
ax2.set_ylim([0,1.1])
ax2.legend()

#%%
tau_d = 8*0.82; p_peak = 0.7
ee_n = p_peak*np.exp(-xx/tau_d)

tau_d = 20*0.82; p_peak = 0.58*0.82
ie_n = p_peak*np.exp(-xx/tau_d)

tau_d = 10*0.82; p_peak = 1*0.82
ei_n = p_peak*np.exp(-xx/tau_d)

tau_d = 20*0.82; p_peak = 1*0.82
ii_n = p_peak*np.exp(-xx/tau_d)
#%%
tau_d = 8; p_peak = 0.7/0.82
ee = p_peak*np.exp(-xx/tau_d)

tau_d = 20; p_peak = 0.58
ie = p_peak*np.exp(-xx/tau_d)

tau_d = 10; p_peak = 1
ei = p_peak*np.exp(-xx/tau_d)

tau_d = 20; p_peak = 1
ii = p_peak*np.exp(-xx/tau_d)
#%%
plt.figure()
plt.plot(xx,ee - ie,'b--')
plt.plot(xx,ei - ii,'y--')

#plt.figure()
plt.plot(xx,ee_n - ie_n,'b')
plt.plot(xx,ei_n - ii_n,'y')
#%%
plt.figure()
plt.plot(xx,(ee - ie)-(ei - ii),'--')
plt.plot(xx,(ee_n - ie_n)-(ei_n - ii_n),'-')

#%%
fig, [ax1,ax2] = plt.subplots(1,2)
tau_d = 7; p_peak = 0.85*0.88
yy = p_peak*np.exp(-xx/tau_d)
ax1.plot(xx,yy, 'b', label='ee_n')

tau_d = 9.0; p_peak = 1*0.82*1.05
yy = p_peak*np.exp(-xx/tau_d)
ax1.plot(xx,yy, 'y', label='ei_n')

tau_d = 20; p_peak = 0.57*0.8
yy = p_peak*np.exp(-xx/tau_d)
ax1.plot(xx,yy, 'g',label='ie_n')

tau_d = 20; p_peak = 1*0.78
yy = p_peak*np.exp(-xx/tau_d)
ax1.plot(xx,yy, 'r',label='ii_n')
ax1.legend()
ax1.set_ylim([0,1.1])


tau_d = 8; p_peak = 0.82
yy = p_peak*np.exp(-xx/tau_d)
ax2.plot(xx,yy,'b--', label='ee')
tau_d = 10; p_peak = 1
yy = p_peak*np.exp(-xx/tau_d)
ax2.plot(xx,yy, 'y--', label='ei')
tau_d = 20; p_peak = 0.58
yy = p_peak*np.exp(-xx/tau_d)
ax2.plot(xx,yy, 'g--',label='ie')
tau_d = 20; p_peak = 1
yy = p_peak*np.exp(-xx/tau_d)
ax2.plot(xx,yy, 'r--',label='ii')
ax2.set_ylim([0,1.1])
ax2.legend()
#%%

ee = 220; tau = 7#8*0.85  ; 
peak = 0.85*0.88 #0.85 * 0.85 
ei = 395; tau = 9.02#10*0.82*1.1 
peak = 1*0.82*1.05

ie = 150; tau = 20#20*0.86*1.15  ; 
peak = 0.57*0.8

ii = 260; tau = 20#20*0.86*1.15  ; 
peak = 1*0.78




