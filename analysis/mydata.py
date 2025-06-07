# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:29:19 2020

@author: nishe
"""
from analysis import firing_rate_analysis
import numpy as np
import scipy.sparse as sparse
import pickle
#%%
class mydata(object):
    '''
    A user defined class used to save data hierarchically.
    Input data musted be either in the format of dictionary or keyword arguments.
    All input dictionaries will be put in a tuple named "param_dict", all input keyword arguments
    will be put in a dictionary named "kwarg".
    If the value of a key in the input dictionary(i.e. "kwarg" and elements in "param_dict") is another dictionary then the data inside that dictionary will
    be loaded recursively in another instance of "mydata".
    '''
        
    def __init__(self, *param_dict, **kwarg):
        
        for dictionary in param_dict:
            for key in dictionary:
                if type(dictionary[key]) is dict:               #If the value of a key in input dictionary is another dictionary,
                    setattr(self, key, mydata(dictionary[key])) #then load the data inside that dictionary recursively in another instance of "mydata".
                else:
                    setattr(self, key, dictionary[key])
        
        for key in kwarg:
            if type(kwarg[key]) is dict:
                setattr(self, key, mydata(kwarg[key]))
            else:
                setattr(self, key, kwarg[key])
    
    def class2dict(self, obj = None):
        '''
        Extract data stored in an instance of a class and save them into a dictionary.
        Data are extracted and saved hierarchically.
        '''
        newdict = {}
        
        if obj is None:
            for key in self.__dict__:
                
                if hasattr(self.__dict__[key], '__dict__'):
                    newdict[key] = self.class2dict(self.__dict__[key])
                else:
                    newdict[key] = getattr(self, key)            
            
        else:            
            for key in obj.__dict__:
                
                if hasattr(obj.__dict__[key], '__dict__'):
                    newdict[key] = self.class2dict(obj.__dict__[key])
                else:
                    newdict[key] = getattr(obj, key)
        
        return newdict
    
#%%    
    def get_sparse_spk_matrix(self, shape, mat_type='csr'):
        
        self.spk_matrix = sparse.coo_matrix((np.ones(len(self.i),dtype=int),(self.i,self.t)), shape=shape)
        if mat_type == 'csr': self.spk_matrix = self.spk_matrix.tocsr()
        if mat_type == 'csc': self.spk_matrix = self.spk_matrix.tocsc()
        
    def get_spike_rate(self, start_time=None, end_time=None,\
                           sample_interval = 1, n_neuron = 4096, window = 10, dt = 0.1, reshape_indiv_rate = True):
            
        indiv_rate = True; popu_rate = False
        save_results_to_input = False                
        spk_rate = firing_rate_analysis.get_spike_rate(self, start_time, end_time, indiv_rate = indiv_rate, popu_rate = popu_rate, \
                               sample_interval = sample_interval, n_neuron = n_neuron, window = window, dt = dt, reshape_indiv_rate = reshape_indiv_rate, save_results_to_input = save_results_to_input)
        self.spk_rate = mydata()
        self.spk_rate.spk_rate = spk_rate
        self.spk_rate.dt = sample_interval
        self.spk_rate.window = window
#%%    
    def get_pop_rate(self, start_time=None, end_time=None,\
                           sample_interval = 1, n_neuron = 3969, window = 10, dt = 0.1):

        indiv_rate = False; popu_rate = True
        reshape_indiv_rate = False
        save_results_to_input = False   
             
        pop_rate = firing_rate_analysis.get_spike_rate(self, start_time, end_time, indiv_rate = indiv_rate, popu_rate = popu_rate, \
                               sample_interval = sample_interval, n_neuron = n_neuron, window = window, dt = dt, reshape_indiv_rate = reshape_indiv_rate, save_results_to_input = save_results_to_input)
        self.pop_rate = mydata()
        self.pop_rate.pop_rate = pop_rate
        self.pop_rate.dt = sample_interval
        self.pop_rate.window = window
#%%    
    def get_centre_mass(self, slide_interval=1, jump_interval=15, detect_pattern=False):        
        
        if not detect_pattern:
            centre_ind, centre, jump_size, jump_dist = firing_rate_analysis.get_centre_mass(self.spk_rate.spk_rate, self.spk_rate.dt, slide_interval, jump_interval, detect_pattern)
        else:
            centre_ind, centre, jump_size, jump_dist, pattern, pattern_size = \
                firing_rate_analysis.get_centre_mass(self.spk_rate.spk_rate, self.spk_rate.dt, slide_interval, jump_interval, detect_pattern)
           
        self.centre_mass = mydata()
        self.centre_mass.centre_ind = centre_ind
        self.centre_mass.centre = centre
        self.centre_mass.jump_size = jump_size
        self.centre_mass.jump_dist = jump_dist
        self.centre_mass.slide_interval = slide_interval
        self.centre_mass.jump_interval = jump_interval
        if detect_pattern:
            self.centre_mass.pattern = pattern
            self.centre_mass.pattern_size = pattern_size
    
    def get_MSD(self, start_time, end_time, \
                   sample_interval = 1, n_neuron = 3969, window = 10, dt = 0.1, \
                   slide_interval = 1, jump_interval=np.array([15]), fit_stableDist = None):
        
        MSD = firing_rate_analysis.get_MSD(self, start_time, end_time, \
                   sample_interval, n_neuron, window, dt,  \
                   slide_interval, jump_interval, fit_stableDist)
        self.MSD = mydata()
        if type(MSD) is not tuple:
            self.MSD.MSD = MSD
        else:
            self.MSD.MSD = MSD[0]
            self.MSD.stableDist_param = MSD[1]
        self.MSD.window = window
        self.MSD.jump_interval = jump_interval
#%%    
    def overlap_centreandspike(self, show_trajectory = False):
        
        self.spk_rate.spk_rate = firing_rate_analysis.overlap_centreandspike(self.centre_mass.centre_ind, self.spk_rate.spk_rate, show_trajectory)
#%%
    def load(self, file_path):
        '''
        load data, the type of data should be in the format of dictionary
        '''
        with open(file_path, 'rb') as file:
            self.__init__((pickle.load(file)))
#%%
    def save(self, data, file_path):
        '''
        save data, the data to be saved is usually in the format of dictionary
        '''
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
            
            
            