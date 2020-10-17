import sys
import numpy as np
import pandas as pd
from IPython import embed
sys.path.insert(0, '../../src')
from trial.sim_datasets.imputed_datasets import sim_dataset
import pickle
from collections import defaultdict

def extract_data(data, variable_stop_index=34): 
    """
    Inputs:
        - data: list of dfs each representing a patient time series with label
        - variable_stop_index: index up until which time series are stored in the columns of the dfs
     Outputs:
        - X: list of patient df's listing only input features 
        - y: list of label 1d time series (as array)
    """     
    X = []
    y = []
    pat_ids = [] 
    for pat in data:
        X.append(
            pat.iloc[:,:variable_stop_index].values
        )  
        y.append(
            pat['SepsisLabel'].values
        )
        pat_ids.append(pat['pat_id'][0]) 
    return np.array(X), np.array(y), np.array(pat_ids)

def data_loader(path):
    """
    Takes path to pickled data file, loads and returns it.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_and_extract_splits(path):
    """
    Loads data from pickle file and extracts the splits (selects used columns from dataframes)
    """
    data = data_loader(path)
    result = defaultdict()
    for key in data.keys():
        result[key] = defaultdict()
        X, y, pat_ids = extract_data(data[key]) 
        result[key]['X'] = X
        result[key]['y'] = y
        result[key]['pat_ids'] = pat_ids 
    return result 

def compute_prevalence(labels):
    """
    Takes list of label arrays (each representing a label series per patient)
    and compute prevalence of positive class (time point wise)
    """
    label_sums = [sum(i) for i in labels]
    label_lens = [len(i) for i in labels]
    case_prev = sum(label_sums) / sum(label_lens) 
    return case_prev


#for testing and inspecting the data simulation, run this script
def main():
    n_vars = 4
    dataset = sim_dataset(0, 10, n_vars) 
    X,y = extract_data(dataset, n_vars)
    embed()

if __name__ == "__main__":
    main() 
