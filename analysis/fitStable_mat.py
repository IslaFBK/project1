#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 17:34:54 2020

@author: shni2598
"""
import numpy as np
import matlab.engine

def fitStable(data, eng=None):
    '''
    fit parameters of stable distribution from input data using Matlab Engine for python;
    input: 
    data: 1-D numpy array
    eng: matlab engine
    output:
    fitparam: 3*4 numpy array; parameters of stable distribution, first row contains four parameters, second and third row contain low and high bound of confidence interval of parameters
    eng: matlab engine
    '''
    if not isinstance(data, np.ndarray):
        raise Exception('Error: data must be numpy array!')
    else:
        if len(data.shape) != 1:
            raise Exception('Error: data must be 1-D numpy array!')
    
    if eng is None:
        eng = start_Mat_Eng()
    
    data = matlab.double(list(data), size=[data.size,1])
    
    eng.workspace['result_fit'] = eng.fitdist(data,'stable')#,nargout=0)
    
    #eng.eval("result_fit=fitdist(jumpsize_mat,'stable')", nargout=0)
    paramci = eng.eval('result_fit.paramci')
    fitparam = eng.eval('[result_fit.alpha result_fit.beta result_fit.gam result_fit.delta]')
    fitparam = np.concatenate((np.array(fitparam),np.array(paramci)),0)
    
    return fitparam, eng

def start_Mat_Eng(args='-nodisplay'):
    ''' start Matlab engine'''
    return matlab.engine.start_matlab(args) # '-nodisplay'

    
            
    
    



