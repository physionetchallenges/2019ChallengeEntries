#!/usr/bin/env python3


'''
BSD 2-Clause License

Copyright (c) 2019, Kaveh Samiee
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import pandas as pd
from keras.models import model_from_yaml
import sys
import os
from calc_ews import q_sofa, sofa, news
from inference import my_kalman_smooth, my_kalman_pred
from base_funcs import read_fex_psv
import csv
import pickle


def get_sepsis_score(test_seq, loaded_model):
    train_col_maxs, train_col_mean, train_col_std = np.load('train_col_stats.npy')
    thresholds = np.load('thresholds.npy')
    rare_params, demograph_col, freq_kalm, paramas_85_inds, paramas_85, base_params_85, LOS_inds = np.load('params.npy', allow_pickle=True)
    with open('columns.pkl', 'rb') as pkl_file:
        cols = pickle.load(pkl_file)

    p_test = read_fex_psv(test_seq, cols)  # 1st row as the column names
    timesteps = 6
    test_features = None
    test_features_pad = None
    skip_kalman = True

    if skip_kalman:
        train_col_mean = train_col_mean[:-8]
        train_col_std = train_col_std[:-8]
    p_test = p_test.fillna(method='ffill')
    p_test = p_test.fillna(method='bfill')
    test_features = np.array(p_test).astype(np.float32)
    inds = np.where(np.isnan(test_features)) 
    test_features[inds] = np.take(train_col_mean, inds[1])
    test_features = np.subtract(test_features, train_col_mean)
    test_features = np.divide(test_features, train_col_std)

    test_seq = np.zeros((test_features.shape[0], timesteps, test_features.shape[1]), dtype='float32')
    for i in range(test_features.shape[0]):
        if i < timesteps:
            test_features_pad = np.pad(test_features, ((timesteps-1, 0), (0, 0)), 'edge')
            test_seq[i, :, :] = np.expand_dims(test_features_pad[i:i+timesteps, :], axis=0)
        else:
            test_seq[i, :, :] = np.expand_dims(test_features[i-timesteps:i, :], axis=0)

    # check for value of scores
    cur_t = test_seq.shape[0] - 1
    ews_scores= 0
    if ((p_test['qsofa'][cur_t] >= 2) or (p_test['sofa'][cur_t] >= 2)
        or (p_test['news'][cur_t] >= 7) or (p_test['mews'][cur_t] >= 4)):
        ews_scores = 1

    ews_scores_d1 = 0
    if ((p_test['qsofa_d1'][cur_t] >= 2) or (p_test['sofa_d1'][cur_t] >= 2)
        or (p_test['news_d1'][cur_t] >= 6) or (p_test['mews_d1'][cur_t] >= 3)):
        ews_scores_d1 = 1

    los = p_test['ICULOS'][cur_t]
    age = p_test['Age'][cur_t]
    
    del p_test
    del test_features
    del test_features_pad

    thr = thresholds[-1]#90
    predictions = loaded_model.predict(test_seq[-1:, :, :])
    predictions_proba = predictions[:, 1]
    predic_labels = (predictions[:, 1] > thr).astype(int)
    
    if age < 18:
        predic_labels = 0 

    if los < 2:
	    predic_labels = ews_scores
    elif los < 4:
        if (predic_labels+ews_scores+ews_scores_d1) > 1:
            predic_labels = 1
    elif predic_labels == 0 and ((ews_scores+ews_scores_d1) > 1):
        predic_labels = 1
		
    if predic_labels and (predictions_proba <= thr):
        predictions_proba =  thr + 0.05
    elif (predic_labels < 1) and (predictions_proba > thr):
        predictions_proba =  thr - 0.05

    return predictions_proba, predic_labels


def load_sepsis_model():
    model_file = 'model.yaml'
    model_wights = 'best_model.hdf5'
    yaml_file = open(model_file, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights(model_wights)

    return loaded_model
