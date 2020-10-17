import pandas as pd
import numpy as np

def pad_binned_data(data, labels, min_pad_length):
    '''
    Inputs: - data: takes a list of binned Dataframes that ONLY contain time series variables (no meta-info, label etc) 
        (each df representing an encounter, list representing eg minibatch)
            - labels: a list of binary arrays
    Returns:
            - data_pad: a padded 3d np array [batch_size, max_batch_len, n_channels] 
            - labels_pad: a padded 2d np array [batch_size, max_batch_len]
            - valid_points: an array listing for each patient the number of valid points to consider down-stream 
    '''
    if len(data) != len(labels):
        raise ValueError("X and y of differing lenghts!")

    batch_size = len(data)
    batch_max_len = np.max([len(pat) for pat in data])
    #make sure that batch is long enough for TCN!
    max_len = np.max([batch_max_len, min_pad_length])
    M = data[0].shape[1] #number of time series channels / medical variables

    #initialize padded output array:
    data_pad = np.zeros([batch_size, max_len, M])
    labels_pad = np.zeros([batch_size, max_len])
    valid_points = np.zeros(batch_size) 
    #loop for filling the array with available data:
    for i in np.arange(batch_size): # over the i-th patient in the batch
        length = len(data[i])
        data_pad[i,:length,:] = data[i]
        labels_pad[i,:length] = labels[i]
        valid_points[i] = length
    return data_pad, labels_pad, valid_points

