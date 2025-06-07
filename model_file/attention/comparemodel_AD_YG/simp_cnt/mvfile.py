#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 12:42:17 2020

@author: shni2598
"""

#import os
#import matplotlib.pyplot as plt
import shutil
#%%
#datapath = '/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/'
#datapath = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/chg_adpt_netsize/'
import glob
#path = '/import/headnode1/shni2598/brian2/NeuroNet_brian/model_file/attention/comparemodel_AD_YG/simp_cnt/data/tmp/dp_num_ie/'
#path = ''
#FilenamesList = glob.glob(path+'*.mp4')
source_path = './good/'
goodsize_dir = './good_size/'
FilenamesList = glob.glob(source_path+'*')
#indList = [None]*len(FilenamesList)
for i in range(len(FilenamesList)):
    filename = FilenamesList[i].split('/')[-1]
    if filename.split('_')[0] == 'p1':
        size = float(filename.split('_')[4][2:])#.split('_')[-1]
    else:
        size = float(filename.split('_')[3][2:])
    if size <= 7.2:
        shutil.copy(FilenamesList[i], goodsize_dir)

# if sys.argv[1] in indList:
#     print('True')
#     sys.exit("Exit, file already exists!")