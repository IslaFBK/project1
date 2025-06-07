# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:43:19 2020

@author: nishe
"""
import numpy as np
#%%
def ignore_shortevent(event, threshold=20):
    '''
    event: an array consists boolean value that indicates if an event happens
    threshold (ms): if the duration of consecutive events(length of consecutive True value) is below threshold then they will be deleted
    onoff: the index when each event starts and ends
    '''    
    dura = 0
    pre_true = 0
    onoff = np.array([[],[]],dtype=int)
    on = 0 
    #off = 0
    for i in range(event.shape[0]):
        if event[i]:
            if pre_true == 0:
                pre_true = 1; dura += 1
                on = i
            else:
                dura += 1
        else:
            if pre_true == 1:
                pre_true = 0
                if dura < threshold:
                    event[i-dura:i] = False
                else:
                    onoff = np.concatenate((onoff, [[on],[i-1]]),1)
                dura = 0
            else:
                continue
            
    return event, onoff
