
import pandas as pd
import numpy as np
import json 
from IPython import embed

#Function to standardize X based on X_train:
def standardize(data=None): #, val=None, test=None,
    #Convert to pd df for easier handling (standardizing with nans, imputing etc)
    #CAVE: if only 1 point in time, its just an 1d array!
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    
    data = pd.DataFrame(data) 

    #Sepsis Challenge provided data statistics (for time series variables)
    mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777,
        60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551 ]) #this line are the statics
    
    std = np.array([
        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
        29.8928, 7.0606,  137.3886, 96.8997, 
        16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367]) #this line are the statics

    #standardize by train statistics:
    data_z = (data-mean)/std

    return data_z


#def impute(pat):
#    #variables = list(pat)[:variable_stop_index]
#    #rest = list(pat)[variable_stop_index:]
#
#    #forward filling variables:
#    pat_imp = pat.ffill()
#    #concat processed variables columns with rest
#    #pat_imp = pd.concat([pat_ff, pat[rest]], axis=1)
#    #replace all remaining nans with 0:
#    pat_imp = pat_imp.replace(np.nan, 0) #replace all remaining nan with 0s
#
#    return pat_imp

def impute(pat, variable_stop_index=34, n_statics=6):
    """
    Impute Time Series and add boolean indicator variable
    """
    series_variables = list(pat)[:variable_stop_index]
    static_variables = list(pat)[variable_stop_index : variable_stop_index + n_statics]
    all_variables = series_variables + static_variables
    indicators = (~pat[series_variables].isnull()).astype(int)
    #forward filling variables:
    pat_ff = pat[series_variables].ffill()
    #concat processed variables columns with rest
    pat_imp = pd.concat([pat_ff, indicators, pat[static_variables] ], axis=1)
    #replace all remaining nans with 0:
    pat_imp = pat_imp.replace(np.nan, 0) #replace all remaining nan with 0s

    return pat_imp


def pad_data_online(data, min_pad_length, batch_size=1):
    '''
    Inputs: - data: takes a list of binned Dataframes that ONLY contain time series variables (no meta-info, label etc)
        (each df representing an encounter, list representing eg minibatch)
    
    Returns:
            - data_pad: a padded 3d np array [batch_size, max_batch_len, n_channels]
            - valid_points: an array listing for each patient the number of valid points to consider down-stream
    '''
    
    batch_max_len = len(data) #since we assume bs=1
    #make sure that batch is long enough for TCN!
    #max_len = np.max([batch_max_len, min_pad_length])
    max_len = batch_max_len
    M = data.shape[1] #number of time series channels / medical variables

    #initialize padded output array:
    data_pad = np.zeros([batch_size, max_len, M])
    valid_points = np.zeros(batch_size)
    #loop for filling the array with available data:
    for i in np.arange(batch_size): # over the i-th patient in the batch
        length = len(data)
        data_pad[i,:length,:] = data
        valid_points[i] = length
    return data_pad, valid_points


def read_configs(path):
    with open(path, 'r') as f:
        return json.load(f)
