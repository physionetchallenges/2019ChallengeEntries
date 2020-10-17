#preprocessing script to binned and imputed final data to apply on simple baselines..

import pandas as pd
import numpy as np 
from IPython import embed
import sys


def impute(pat, variable_stop_index=34):
    variables = list(pat)[:variable_stop_index]
    rest = list(pat)[variable_stop_index:]
    
    #forward filling variables:
    pat_ff = pat[variables].ffill()
    #concat processed variables columns with rest
    pat_imp = pd.concat([pat_ff, pat[rest]], axis=1)
    #replace all remaining nans with 0:
    pat_imp = pat_imp.replace(np.nan, 0) #replace all remaining nan with 0s
    
    return pat_imp


def binning(): 
    pass
