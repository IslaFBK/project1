#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:27:28 2020

@author: shni2598
"""
import sys
import post_analysis as psa
import pickle
#%%

class analyse:
    
    def __init__(self, timepath, indexpath, file_num, area_num):
        
        #for i in range(1,area+1):
        
        attrname = ['spka%d'%i for i in range(1,area_num+1)]
            
        for i in range(len(attrname)):
            setattr(self, attrname[i], psa.importmatdata(timepath, indexpath, file_num, area=i+1))
            #spka1 = psa.importmatdata(timepath, indexpath, file_num, area)


#print(sys.argv[:])        
    
#%%
#area = 2
#['a%d'%i for i in range(1,area+1)]
#timepath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli/timev%d%d.mat'#%(area,file_num)
#indexpath = '/import/headnode1/shni2598/brian2/data/twoarea_stimuli/indexv%d%d.mat'#%(area,file_num)
#
#data = analyse(timepath, indexpath, file_num = 0, area_num = 2)

if __name__ == '__main__':
    
    #print(sys.argv[:])
    #print(int(sys.argv[3]))
    data = analyse(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    
    with open('data2','wb') as file:
        pickle.dump(data,file)

