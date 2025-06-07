#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:36:03 2021

@author: shni2598
"""

#%%
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/2to1/'

data_analy = mydata.mydata()

loop_num = 100
data_analy.load(datapath+'data_anly%d.file'%loop_num)
#%%
n_ie = 81#50; 
trial_per_ie = 10
n_amp_stim = 6
n_per_amp = 10
n_amp_stim_att = 3

data_analy.hz_loc

#%%
plt.figure()
plt.plot(data_analy.hz_loc[:,:,0].reshape(3,10,-1).mean(1)[0])

#%%



#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/2to1/benmrk_onearea/'
datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/two_area/data/2to1/'

data = mydata.mydata()

loop_num = 79
data.load(datapath+'data%d.file'%loop_num)
#%%

first_stim = 9; last_stim = 10
start_time = data.a1.param.stim.stim_on[first_stim,0] - 200
end_time = data.a1.param.stim.stim_on[last_stim,0] + 400

data.a1.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a1.param.Ne, window = 10)
data.a1.ge.get_centre_mass()
data.a1.ge.overlap_centreandspike()

data.a2.ge.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ne, window = 10)
data.a2.ge.get_centre_mass()
data.a2.ge.overlap_centreandspike()

#data.a1.gi.get_spike_rate(start_time=start_time, end_time=end_time, sample_interval=1, n_neuron = data.a2.param.Ni, window = 10)
#frames = int(end_time - start_time)
frames = data.a1.ge.spk_rate.spk_rate.shape[2]

stim_on_off = data.a1.param.stim.stim_on-start_time
stim_on_off = stim_on_off[stim_on_off[:,0]>=0][:int(last_stim-first_stim)+1]
stim = [[[[31,31]], [stim_on_off], [[6]*stim_on_off.shape[0]]], None]
adpt = [None, [[[31,31]], [[[0, data.a1.ge.spk_rate.spk_rate.shape[-1]]]], [[6]]]]
ani = firing_rate_analysis.show_pattern(spkrate1=data.a1.ge.spk_rate.spk_rate, spkrate2=data.a2.ge.spk_rate.spk_rate, \
                                        frames = frames, start_time = start_time, interval_movie=10, anititle='',stim=stim, adpt=adpt)


#%%

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
#%%
import gc
import sys

def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz
#%%

print(get_obj_size(data.class2dict()))


#%%
dct = {'a': 5, 'b': 7}

print(sys.getsizeof(dct))


#%%
def findsize(dicti):
    size = 0
    for key in dicti.keys():
        #if dicti[key] is dict
        if isinstance(dicti[key],dict):
            size += findsize(dicti[key])
        else:
            if isinstance(dicti[key], np.ndarray):
                print(key+':',dicti[key].nbytes)
                size += dicti[key].nbytes
            else:
                print(key+':',sys.getsizeof(dicti[key]))
                size += sys.getsizeof(dicti[key])
    return size
#%%
print(findsize(data.class2dict())/1000000, 'MB')









