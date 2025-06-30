import brian2.numpy_ as np
from sklearn.linear_model import LinearRegression
from levy import fit_levy
from myscript.ie_search.compute_MSD_pdx import compute_MSD_pdx
from myscript.ie_search.utils import is_param_near
from myscript.ie_search.batch_repeat import batch_repeat

import logging
logging.getLogger('brian2').setLevel(logging.WARNING)


'''
Criterion of critical or not, and search till the critical states
'''

