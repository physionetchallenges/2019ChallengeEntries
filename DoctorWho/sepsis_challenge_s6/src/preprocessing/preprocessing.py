# author: Michael Moor 2019

'''
##TODO: 
Steps:
1.  Gathering train/val/test in df:
    Given a split i:
    Gather all patient data of train / val / test each in a concatenated pd dataframe

2.  Standardizing:
    Using train df -> get statistics per feature and apply to all 3 (train/val/test)

(3.  Processing of static features)      

4.  Compact Transform:
    Once time series are standardized, apply compact transform and pickle resulting data for quick access
'''
import numpy as np
import pandas as pd
import glob
from IPython import embed
import sys 
import time
import os
import pickle
from collections import defaultdict

from util import standardize, compact_transform
from bin_and_impute import impute

def read_and_process_files(ids=None, path = "../../../../data/extracted/", imputation_flag=True):
    result = [] #pd.DataFrame() 
    for i, pat_id in enumerate(ids):
        #if i > 4:  # for quick debugging
        #    return result
        datapath = path + 'p' + str(pat_id).zfill(6) + '.psv'
        #print(datapath) #format: ../../../../data/extracted/p109230.psv
        pat = pd.read_csv(datapath, sep='|')
        pat['pat_id'] = pat_id #create column for pat_id
        #standardize (before imputation) 
        pat, _ = standardize(pat)
        #impute (if requested)
        if imputation_flag:
            pat = impute(pat, variable_stop_index=34)
            if pat.isnull().sum().sum() > 0:
                print(f'Patient {pat_id}: Nan remaining after preprocessing!') 
        result.append(pat) 
    return result

#hard-coded path for prototyping:
def preprocessing(split=0, compact=False, path = "../../../../data"):
    datapath = path + "/extracted/*.psv"
    split_file = path + "/split_info.pkl"
    
    #check if splits exists and read pkl file, throw error message otherwise
    if not os.path.isfile(split_file):
        raise ValueError('Splits do not exist yet, compute them first with scripts/create_splits.py..') 
    else:
        with open(split_file, "rb") as f:
            info = pickle.load(f)
    #info['split_0']['train'] 
    current_split = f'split_{split}'
    
    #Gather train/val/test splits for preprocessing:
    datasplit = defaultdict()
    #Loop over train/val/test data: 
    for subset in ['train', 'validation', 'test']:
        print(f'Processing {subset} split...')
        ids =  info[current_split][subset]
        
        #read single patient files and append dfs into list of per-patient dfs
        data = read_and_process_files(ids)
        #apply compact transform for MGP Adapter model 
        if compact: #default false
            print('Converting to compact format..')
            data_c = compact_transform(data_z)    
            datasplit[subset] = data_c 
        else: #default true
            datasplit[subset] = data 
 
    #Dump preprocessed split:
    outpath = path + '/mgp_tcn_data'
    if not os.path.exists(outpath):
        print('make dir for outpath..')
        os.makedirs(outpath)
    
    print('Dumping processed split...')
    if compact:
        with open(outpath + f'/compact_data_split_{split}.pkl', 'wb') as f:
            pickle.dump(datasplit, f)
    else:
        with open(outpath + f'/imputed_data_split_{split}.pkl', 'wb') as f:
                pickle.dump(datasplit, f)
    return datasplit

#5 static variables: 
'''
 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime'
'''
#34 time series variables: 
'''
HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
'Fibrinogen', 'Platelets'
'''

if __name__ == "__main__":
    preprocessing()


