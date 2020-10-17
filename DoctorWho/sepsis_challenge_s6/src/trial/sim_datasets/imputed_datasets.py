'''
Script that contains dataset code:
- simulation dataset
- 
'''
import numpy as np
from collections import defaultdict
import pandas as pd 

#---------------------------------
# 1. CODE FOR SIMULATION DATASET:
#---------------------------------

# generate list of pd dataframes containing all patient data: M feature columns, 1 label column
 
def sim_dataset(rs, num_encs,M,pos_class_rate = 0.5):
    """
    Returns everything we need to run a model requiring pre-imputed data.
    """
    np.random.seed(seed=rs)
    data = []
    num_timepoints = np.random.randint(30,50, size=num_encs)
    #signal used to modify timeseries of cases:
    channel_vec = np.random.randint(-1,2,M) #vector of values from -1 to 1 of length M 
    #Define patient ids and cases & controls 
    pat_ids = np.arange(num_encs)
    case_index = int(num_encs*pos_class_rate) 
    case_ids = pat_ids[:case_index]
    control_ids = pat_ids[case_index:] 
   
    print(f'Simming {num_encs} patients ..') 
    #Generate Data for cases and controls
    for i in pat_ids:
        length = num_timepoints[i]
        if i < case_index:
            #generate case
            labels, onset = create_label(length, case=True)
            X = generate_time_series(length, M)
            X = add_signal(X, onset, channel_vec)  
            X['SepsisLabel']= labels 
        else:
            #generate control
            labels, _ = create_label(length, case=False)
            X = generate_time_series(length, M)
            X['SepsisLabel']= labels
        data.append(X) 
    #Shuffle list of patients
    np.random.shuffle(data)
    return data

def create_label(length, case=False):
    onset = None
    #label vector: 1s after onset, 0 otherwise
    labels = np.zeros(length)
    if case: #set onset
        #pick random onset time during patient stay
        onset = np.random.randint(1, length)                
        labels[onset:] = 1
    return labels, onset
    
def generate_time_series(length, M):
    """
    generate time series of length 'length' and of 'M' channels
    """
    #standard normal values
    X = np.random.normal(0,1,[length,M])
    return pd.DataFrame(X)
    
def add_signal(X, onset, channel_vec):
    """
    adds signal to case time series: perturbs ts from onset on, 
        channel_vec determines how channels are affected.
    """
    def sigmoid(z, a, b):
        return 1 / (1 + np.exp(-b*(z-a))) 
    signal_len = len(X)
    srange = np.arange(signal_len)
    signal = sigmoid(srange, onset, np.sqrt(1/signal_len) )
    X = X.mul(signal, axis=0) 
    X[onset:] = X[onset:] * channel_vec 
    return X


