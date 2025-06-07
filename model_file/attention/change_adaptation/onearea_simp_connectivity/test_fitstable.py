#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:35:41 2020

@author: shni2598
"""

from __future__ import print_function
import fitstable
import matlab
#%%
my_fitstable = fitstable.initialize()

inputIn = matlab.double([-7.0, -5.0, -3.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 5.0, 10.0], size=(16, 1))
paramsOut = my_fitstable.fitstable(inputIn)
print(paramsOut, sep='\n')

my_fitstable.terminate()