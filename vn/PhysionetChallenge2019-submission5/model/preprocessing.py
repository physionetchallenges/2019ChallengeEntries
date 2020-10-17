import numpy as np
import os
import tensorflow as tf
from utils import *

flags = tf.app.flags.FLAGS


def extract_data(Data):
  # Extraction training set from data to train by point
  idx_sepsis, idx_normal = [], []	
  for j in Data.keys():
		# j : pXXXXXXXXX
    data = Data[j]
    for i in range(len(data)):
      # i: index
      if data[i][-2] == 0: 
        idx_normal.append([j,i])
      else:	
        idx_sepsis.append([j,i])
  return idx_sepsis, idx_normal

def padding(inputs, size):
    T = flags.input_length
    tmp = np.zeros([T, size[1]+1])
    inputs = inputs[-T:,:]
    
    input_shape = np.asarray(inputs.shape)
    tmp[tmp.shape[0]-input_shape[0]:,:-1] = inputs
    tmp[tmp.shape[0]-input_shape[0]:, -1] = 1 # mask column 
    return tmp 

def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:]
        values = values[:, :]
    return (values, column_names)

def get_mask(data):
    mask = (np.isnan(data[:,:-6])==False)*1
    return mask

def get_delta(mask):
    delta = np.zeros_like(mask)
    for j in range(len(mask)):
        if j != 0:
            index1 = np.where(mask[j]==1)[0]
            index0 = np.where(mask[j]==0)[0]
            delta[j][index1] = 1
            delta[j][index0] = delta[j-1][index0]+1
    return delta

def forward_fill(data):
    for j in range(len(data)):
        index_nan = np.where(np.isnan(data[j])==True)[0]
        #print(index_nan)
        data[j,index_nan] = data[j-1,index_nan]
        #data[j,index_nan] = np.mean(data[1:j-1,index_nan])
    return data

def zero_fill(data):
    for j in range(len(data)):
        index_nan = np.where(np.isnan(data[j])==True)[0]
        data[j][index_nan] = 0
    return data

def median_fill(data):
    dataset = {'A1':0, 'A2':1, 'A3':2, 'A4':3, 'A5':4, \
								'B1':5, 'B2':6, 'B3':7, 'B4':8, 'B5':9}
    training_set = flags.training_set.split('/')
    moments_set = []
    for i in training_set:
        moments_set.append(dataset[i])
    
    medians = get_median()
    median = np.mean(medians[moments_set], 0)
    for j in range(len(data)):
        index_nan = np.where(np.isnan(data[j])==True)[0]
        data[j][index_nan] = median[index_nan]
    return data

def get_moments(dataset, type=0):
    if type == 0:
      means, stds = moments_bank0()
    elif type == 1:
      means, stds = moments_bank1()
    elif type == 2:
      means, stds = moments_bank2()
    elif type == 3:
      means, stds = moments_bank3()
    elif type == 5:
      means, stds = moments_bank5()
    mean = np.mean(means[dataset], 0)
    std = np.mean(stds[dataset], 0)
    return mean, std

def get_median():
    median = np.array([[ 8.384e+01,  9.738e+01,  3.694e+01,  1.191e+02,  7.753e+01,
             5.923e+01,  1.831e+01,     np.nan, -9.351e-02,  2.430e+01,
             4.919e-01,  7.390e+00,  4.064e+01,  9.217e+01,  8.921e+01,
             2.145e+01,  9.999e+01,  8.308e+00,  1.056e+02,  1.173e+00,
             2.208e+00,  1.276e+02,  1.841e+00,  2.008e+00,  3.535e+00,
             4.143e+00,  1.532e+00,  9.405e+00,  3.169e+01,  1.079e+01,
             3.427e+01,  1.156e+01,  3.014e+02,  2.115e+02,  6.277e+01,
             5.760e-01,  4.933e-01,  5.067e-01, -4.643e+01,  2.077e+01],
           [ 8.395e+01,  9.741e+01,  3.694e+01,  1.194e+02,  7.765e+01,
             5.939e+01,  1.832e+01,     np.nan, -2.091e-01,  2.424e+01,
             4.953e-01,  7.389e+00,  4.049e+01,  9.196e+01,  8.625e+01,
             2.149e+01,  1.001e+02,  8.315e+00,  1.057e+02,  1.178e+00,
             2.225e+00,  1.277e+02,  1.845e+00,  2.014e+00,  3.546e+00,
             4.140e+00,  1.561e+00,  8.977e+00,  3.166e+01,  1.079e+01,
             3.420e+01,  1.159e+01,  3.034e+02,  2.088e+02,  6.285e+01,
             5.779e-01,  4.929e-01,  5.071e-01, -4.787e+01,  2.073e+01],
           [ 8.397e+01,  9.740e+01,  3.694e+01,  1.198e+02,  7.785e+01,
             5.945e+01,  1.838e+01,     np.nan, -2.155e-01,  2.427e+01,
             4.962e-01,  7.389e+00,  4.043e+01,  9.196e+01,  8.600e+01,
             2.152e+01,  9.889e+01,  8.330e+00,  1.056e+02,  1.185e+00,
             2.129e+00,  1.277e+02,  1.847e+00,  2.015e+00,  3.551e+00,
             4.138e+00,  1.491e+00,  9.548e+00,  3.170e+01,  1.080e+01,
             3.423e+01,  1.152e+01,  3.055e+02,  2.086e+02,  6.270e+01,
             5.813e-01,  4.921e-01,  5.079e-01, -4.727e+01,  2.080e+01],
           [ 8.399e+01,  9.742e+01,  3.694e+01,  1.198e+02,  7.789e+01,
             5.951e+01,  1.835e+01,     np.nan, -2.169e-01,  2.426e+01,
             4.966e-01,  7.390e+00,  4.044e+01,  9.195e+01,  8.637e+01,
             2.157e+01,  9.928e+01,  8.333e+00,  1.056e+02,  1.184e+00,
             2.173e+00,  1.277e+02,  1.849e+00,  2.014e+00,  3.541e+00,
             4.137e+00,  1.466e+00,  9.061e+00,  3.172e+01,  1.081e+01,
             3.419e+01,  1.155e+01,  3.083e+02,  2.080e+02,  6.262e+01,
             5.817e-01,  4.941e-01,  5.059e-01, -4.710e+01,  2.078e+01],
           [ 8.398e+01,  9.742e+01,  3.694e+01,  1.199e+02,  7.791e+01,
             5.949e+01,  1.834e+01,     np.nan, -2.390e-01,  2.426e+01,
             4.966e-01,  7.389e+00,  4.039e+01,  9.203e+01,  8.534e+01,
             2.170e+01,  1.011e+02,  8.337e+00,  1.056e+02,  1.190e+00,
             2.221e+00,  1.278e+02,  1.839e+00,  2.015e+00,  3.548e+00,
             4.138e+00,  1.480e+00,  8.876e+00,  3.174e+01,  1.081e+01,
             3.422e+01,  1.154e+01,  3.100e+02,  2.085e+02,  6.262e+01,
             5.819e-01,  4.942e-01,  5.058e-01, -4.868e+01,  2.085e+01]])    
    return median

def get_minmax():
    bound = {'HR':[153, 17], # 60-100
                 'O2Sat':[100, 85], # 95-100
                 'Temp':[40, 33], # 36.1-37.2, >38: serious infection
                 'SBP':[207, 35], 
                 'MAP':[139, 18], 
                 'DBP':[110, 9], 
                 'Resp':[40, 5], # 12-20(25)
                 'EtCO2':[9, 65],
                 'BaseExcess':[17, -17], # -2-2
                 'HCO3':[42, 6], # 22-28
                 'FiO2':[1.0, 0.05], 
                 'pH':[7.7, 7.0], # 7.38-7.42
                 'PaCO2':[77, 6], # 38-42
                 'SaO2':[100, 50], # 94-100
                 'AST':[1000, 0], # 10-40
                 'BUN':[105, 0], # 7-20
                 'Alkalinephos':[700, 0], # 44-147
                 'Calcium':[12, 5], # 8.5-10.5
                 'Chloride':[130, 82], # 97-107
                 'Creatinine':[8, 0], # 0.5-1.5
                 'Bilirubin_direct':[21, 0], # <0.3
                 'Glucose':[340, 0], # 4-5.5(7.8 after eating)
                 'Lactate':[12, 0], # 0.5-1.0 (less than 2 for patients)
                 'Magnesium':[3.6, 0.5], # 1.5-2.5
                 'Phosphate':[9.4, 0], # 2.5-4.5
                 'Potassium':[6.7, 1.6], # 3.5-5.0
                 'Bilirubin_total':[24, 0], # 0.1-1.2
                 'TroponinI':[55, 0], # <0.04, >0.4:cardiac injury
                 'Hct':[50, 11], # 45-52 for men, 37-48 for women
                 'Hgb':[18, 3], # 13.5-17.5 for men, 12.0-15.5 for women
                 'PTT':[137, 0], # 30-45
                 'WBC':[43, 0], # 4.5-11
                 'Fibrinogen':[926, 0], # 150-400
                 'Platelets':[636, 0] # 150-450
                 }  
    index = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']  
    maximum, minimum = [], []
    for i in index:
        maximum.append(bound[i][0])
        minimum.append(bound[i][1])
    return np.asarray(maximum), np.asarray(minimum)

def standardization(data, col, type):
    # A1-A5 = 0:4, B1-B5 = 5:9
    dataset = {'A1':0, 'A2':1, 'A3':2, 'A4':3, 'A5':4, \
								'B1':5, 'B2':6, 'B3':7, 'B4':8, 'B5':9}
    training_set = flags.training_set.split('/')
    moments_set = []
    for i in training_set:
        moments_set.append(dataset[i])
    mean, std = get_moments(moments_set, type)
    data = (data-mean[col])/std[col]
    index_outlier = np.abs(data) > 5 
    data[index_outlier] = 5 * np.sign(data[index_outlier])
    return data

def standardization_physionet(data):
    means, stds = moments_bank10()
    data = (data-means)/stds
    return data

def coxbox(data, col):
    lmbdas = {0: 0.35286,
					 1: 17.753,
					 2: 3.6693,
					 3: -0.0071159,
					 4: 0.065656,
					 5: -0.025092,
					 6: 0.21117,
					 7: 1,
					 8: 0.94150,
					 9: 1.0256,
					 10: 0.020988,
					 11: 18.536,
					 12: -0.11490,
					 13: 10.222,
					 14: -0.25527,
					 15: -0.11019,
					 16: -0.40021,
					 17: 0.40837,
					 18: 1.4743,
					 19: -0.47565,
					 20: 0.0075555,
					 21: -0.17691,
					 22: -0.36232,
					 23: 0.15957,
					 24: 0.17189,
					 25: -0.27963,
					 26: -0.23757,
					 27: 0.044365,
					 28: 0.18826,
					 29: 0.27509,
					 30: -1.3853,
					 31: 0.34259,
					 32: -0.029572,
					 33: 0.30373,
					 34: 1.7906,
					 35: 0.0513820,
					 36: 0.0046261,
					 37: -0.0046261}
    data = (data**lmbdas[col] - 1)/lmbdas[col]
    return data

def remove_outlier(data):
    maximum, minimum = get_minmax()
    for i in range(34):
        idx_max = data[:,i] > maximum[i]
        idx_min = data[:,i] < minimum[i]
        data[idx_max,i] = maximum[i]
        data[idx_min,i] = minimum[i]
    return data

def transform_dist(data):
    if flags.preprocessing == 0:
        transform_index = [1,1,1,1,1,1,1,1,
													1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    elif flags.preprocessing == 3:
        transform_index = [5,5,5,5,5,5,5,5,\
		 										5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
        
    mx, mn = get_minmax()
    for i in range(34):    
        if transform_index[i] == 1:  # remove outlier
            data[:,i] = standardization(data[:,i], i, 0)
        elif transform_index[i] == 5:
            data[:,i] = coxbox(data[:,i], i)
            data[:,i] = standardization(data[:,i], i, 5)
    return data

def to_onehot(i, depth):
    temp = np.zeros(depth, dtype=np.float32)
    temp[i] = 1
    return temp

def data_augmentation(data):
    ddata = np.zeros_like(data[:,:34])
    for j in range(1, len(data)):
        ddata[:,:34][j] = data[:,:34][j]-data[:,:34][j-1]
    return ddata

def merge_tables(datas):
    return np.concatenate([datas[j] for j in range(len(datas))], axis=1)

def Preprocessing(values, columns):
    data = values[:,:40]
    data = remove_outlier(data)
    mask = get_mask(data)
    delta = get_delta(mask)

    data = standardization_physionet(data)
    data = zero_fill(data)
    data = forward_fill(data)
    ddata = data_augmentation(data)
    data_tot = merge_tables([delta, mask, ddata, data])	
    return padding(data_tot, data_tot.shape)

