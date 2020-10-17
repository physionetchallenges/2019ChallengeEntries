#!/usr/bin/env python

import numpy as np
import pickle
import copy
def transform_data(data):
    # transformation
    log_trans = [14, 15, 16, 17, 19, 20, 22, 23, 25, 26, 27, 30, 31, 39]
    nlog_trans = [38]
    square_trans = [1, 13]
    def log10(A):
        return [np.log10(np.float(item)) if not np.isnan(item) else item for item in A]

    def nlog10(A):
        return [np.log10(-np.float(item) + 0.00001) if not np.isnan(item) else item for item in A]

    def power2(A):
        return [np.power(np.float(item), 2) if not np.isnan(item) else item for item in A]

    patient_i = np.array(data)
    patient_i[:,log_trans] = map(log10, patient_i[:,log_trans])
    patient_i[:,nlog_trans] = map(nlog10, patient_i[:,nlog_trans])
    patient_i[:,square_trans] = map(power2, patient_i[:,square_trans])
    data = patient_i
    return data
def get_sepsis_score(data, model):
    model, mean_train, std_train, threshold = model

    data_ = copy.deepcopy(data)
    data_ = transform_data(data_)
    patient_i = np.array(data_)
    p_shape = patient_i.shape
    # impute missing values of test set
    for j in range(p_shape[0]):  # time_step
        for k in range(p_shape[1]):  # features
            if j == 0:
                if np.isnan(patient_i[0][k]):
                    data_[0][k] = mean_train[k]
            else:
                if np.isnan(patient_i[j][k]):
                    data_[j][k] = data_[j - 1][k]
    # standardization
    data_ = np.array(data_, np.float32) - mean_train
    data_ = data_ / std_train

    win_size = 6
    t_samp = data_[-win_size:]
    t_samp = t_samp.flatten()
    # padding with zeros
    max_length = win_size * 40
    if max_length > len(t_samp):
        t_samp = np.concatenate((np.full((max(max_length - len(t_samp), 0)), 0), t_samp))

    score = model.predict_proba(np.expand_dims(t_samp,0))[:, 1][0]

    label = score >= threshold

    return score, label.astype(int)

def load_sepsis_model():
    fpath = './model_data/model_weights.dat'
    # load model from
    model = pickle.load(open(fpath, "rb"))
    print ("Model ({}) loaded.".format(fpath))

    # load  mean, variance and threshold
    fpath = './model_data/model_mean_std_train.npz'
    with np.load(fpath) as f:
        mean = f['mean_train']
        std = f['std_train']
        # threshold = f['sensitivity85_thresholds']

    # fpath = './model_data/model_ATU.npz'
    # with np.load(fpath) as f:
    #     threshold = f['threshold']
    threshold = 0.022006932646036148
    return [model, mean, std, threshold]
