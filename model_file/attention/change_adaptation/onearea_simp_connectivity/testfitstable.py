#!/usr/bin/env python
"""
Sample script that uses the fitstable module created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

from __future__ import print_function
import fitstable
import matlab

my_fitstable = fitstable.initialize()

inputIn = matlab.double([-7.0, -5.0, -3.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 3.0, 5.0, 10.0], size=(16, 1))
paramsOut = my_fitstable.fitstable(inputIn)
print(paramsOut, sep='\n')

my_fitstable.terminate()

#%%
#direct import
import importlib
import sys
#%%
file_path = '/import/headnode1/shni2598/anaconda3/envs/brian/lib/python3.7/site-packages/matlab/engine/__init__.py'#tokenize.__file__
module_name = 'engine'#tokenize.__name__

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)










