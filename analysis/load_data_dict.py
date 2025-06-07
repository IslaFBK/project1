# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:04:33 2020

@author: nishe
"""

class data_onegroup:
    def __init__(self, param_dict):
        for key in param_dict:
            setattr(self, key, param_dict[key])    

class data_onearea:
    def __init__(self, param_dict):
        for key in param_dict:
            if key == 'param':
                setattr(self, key, param_dict[key])
            else:
                setattr(self, key, data_onegroup(param_dict[key]))  
        
class data_multiarea:
    def __init__(self, param_dict):
        for key in param_dict:
            if key == 'param':
                setattr(self, key, param_dict[key])
            else:
                setattr(self, key, data_onearea(param_dict[key]))

  