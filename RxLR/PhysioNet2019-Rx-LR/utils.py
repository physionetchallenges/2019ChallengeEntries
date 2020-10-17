#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:47:34 2019

@author: edward
"""

import os
import numpy as np
from shutil import copyfile

# Read data from one psv file (one patient) into a dictionary, with each key being
# one of the columns in the psv file except for columns that don't reflect time-series
# data (e.g. age, gender)
def read_patient_data(input_file):
    
    with open(input_file, 'r') as f:
        headings = f.readline().strip().split('|')    
        data = {var: [] for var in headings}
           
        for line in f:
            values = line.strip().split('|')
            
            # Replace nan values with np.nan and values convert to float
            values = [np.nan if x == 'NaN' else float(x) for x in values]
            
            for j, var in enumerate(headings):
                data[var].append(values[j])

        # Demographics (not time-series)
        # Take only one value, no need for list
        data['Age'] = int(data['Age'][0])
        data['Gender'] = 'Female' if data['Gender'][0] == '0' else 'Male'
        data['HospAdmTime'] = float(data['HospAdmTime'][0])
        data['HasSepsis'] = int(max(data['SepsisLabel']))
        
        data['SepsisLabel'] = [int(x) for x in data['SepsisLabel']]
            
    return data


def read_all_patients(directory):
    file_names = os.listdir(directory)
    
    all_patients = {}
    
    for x in file_names:
        all_patients[x] = read_patient_data(directory + '/' + x)
        
    return all_patients

# Return a subset of the input dictionary that has patients 
# with sepsis (SepsisLabel = 1 at some point)
def get_sepsis_patients(dictionary):
    return {key: value for key, value in dictionary.items() if value['HasSepsis'] == 1}

def get_no_sepsis(dictionary):
    return {key: value for key, value in dictionary.items() if value['HasSepsis'] == 0}



# Create label psv
def create_label_psv(input_file, output_file):
    data = read_patient_data(input_file)
    with open(output_file, 'w') as f:
        f.write('SepsisLabel\n')
        for x in data['SepsisLabel']:
            f.write(str(x) + '\n')

# Create directory with files for testing
# Usage example: 
# create_test_dir(test_key, "../testing", "../training")
def create_test_dir(test_keys, test_dir, training_dir):
    for name in test_keys:
        copyfile(training_dir+"/"+name, test_dir+"/"+name)
        

# https://stackoverflow.com/questions/13728392/moving-average-or-running-mean       
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    