#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 12:31:17 2021

@author: shni2598
"""


#%%
plt.figure()
plt.scatter(data.a1.param.e_lattice[:,0], data.a1.param.e_lattice[:,1], s=0.5)
plt.scatter(data.a1.param.i_lattice[:,0], data.a1.param.i_lattice[:,1], s=0.5)
#%%
plt.figure()
plt.scatter(data.a2.param.e_lattice[:,0], data.a2.param.e_lattice[:,1], s=0.5)
plt.scatter(data.a2.param.i_lattice[:,0], data.a2.param.i_lattice[:,1], s=0.5)
#%%
plt.figure()
plt.scatter(data.a1.param.e_lattice[:,0], data.a1.param.e_lattice[:,1], s=0.5)

for n in range(len(n_in_bin)):

    plt.scatter(data.a1.param.e_lattice[n_in_bin[n]][:,0], data.a1.param.e_lattice[n_in_bin[n]][:,1], s=1.5, marker='^')


#%%
posi_stim_e1 = NeuronGroup(4096, \
                        '''rates =  bkg_rates + stim_1*scale_1(t) + stim_2*scale_2(t) : Hz
                        bkg_rates : Hz
                        stim_1 : Hz
                        stim_2 : Hz
                        ''', threshold='rand()<rates*dt')

posi_stim_e1.bkg_rates = 0*Hz
posi_stim_e1.stim_1 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[0, 0]])*Hz
posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, -32]])*Hz
#posi_stim_e1.stim_2 = psti.input_spkrate(maxrate = [200], sig=[6], position=[[-32, 0]])*Hz

#%%
ipt_rate = (posi_stim_e1.stim_1/Hz).reshape(64,64) + (posi_stim_e1.stim_2/Hz).reshape(64,64)
plt.matshow(ipt_rate)


#%%

from scipy.special import iv
#%%
iv(0,2)
#%%
A = 2;
P = 1
st1 = 0
st2 = 32*2**0.5
pos = np.arange(-32, 25, 1)*2**0.5

curv1 = A*np.exp(P*np.cos((pos-st1)/(32*2**0.5/np.pi)))/(2*np.pi*iv(0, P))
curv2 = A*np.exp(P*np.cos((pos-st2)/(32*2**0.5/np.pi)))/(2*np.pi*iv(0, P))
#%%
plt.figure()
plt.plot(curv1)
plt.plot(curv2)


#%%
def response(x, A, P, pow_n, c50, r_max):
    '''
    phenomenological normalization model (Busse et al. 2009)
    '''
    st1 = 0
    st2 = 32*2**0.5

    curv1 = A*np.exp(P*np.cos((x[:-2]-st1)/(32*2**0.5/np.pi)))/(2*np.pi*iv(0, P))
    curv2 = A*np.exp(P*np.cos((x[:-2]-st2)/(32*2**0.5/np.pi)))/(2*np.pi*iv(0, P))
    
    return r_max*(curv1*x[-2]**pow_n + curv2*x[-1]**pow_n)/(c50**pow_n + np.sqrt(x[-2]**2+x[-1]**2)**pow_n)
#%%
def _wrap_func(func, xdata, ydata, transform):
    if transform is None:
        def func_wrapped(params):
            return func(xdata, *params) - ydata
        
res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)

def leastsq(func, x0, args=(), Dfun=None, full_output=0,
            col_deriv=0, ftol=1.49012e-8, xtol=1.49012e-8,
            gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    
shape, dtype = _check_func('leastsq', 'func', func, x0, args, n)

def _check_func(checker, argname, thefunc, x0, args, numinputs,
                output_shape=None):
    res = atleast_1d(thefunc(*((x0[:numinputs],) + args)))
    
