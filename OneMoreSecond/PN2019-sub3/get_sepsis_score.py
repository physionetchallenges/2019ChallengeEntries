#!/usr/bin/env python

import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from keras import layers
from keras import models
from sklearn import metrics
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from keras.models import Sequential

from keras import backend as K

# parameters


org_feature = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
               'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
               'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
               'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
               'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
               'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
               'HospAdmTime', 'ICULOS']

#7
custom = ['HR','Temp', 'Resp','SBP', 'DBP', 'O2Sat',  'ICULOS']
feature_index = [0,2,6,3,5,1,39]

# xgb-preclassifer
xgb_preclassifer = xgb.Booster(model_file='pre_xgb.model') 
feature_xg=['HR','O2Sat', 'SBP', 'DBP', 'Resp','ICULOS', 'HospAdmTime','Unit1', 'BaseExcess', 'Age', 'MAP', 'Alkalinephos', 'AST']
xgb_t = 0.5

def test(test_data):
    dtest = xgb.DMatrix(test_data[feature_xg])
    preds = xgb_preclassifer.predict(dtest)
    preds = [1 if p > xgb_t else 0 for p in preds]

    return 1 if sum(preds) >= 2 else 0

# LSTM
# change me everytime you change the model
NOW = '2019-08-08-16-47-08'
model_name = f'LSTM_{NOW}.h5'
LSTM_t = 0.3
cut = 100

def feature_engineer_cut(train):
    if len(train) < cut:
        train = np.concatenate((train,[train[-1] * 0 - 1] * (cut - len(train))), axis = 0)
        return train.copy()
    elif len(train) >= cut:
        return train[:cut].copy()
    
def feature_fill_helper(data,index):
    col_fill=data[:,index]
    col_fill=col_fill.reshape(1,data.shape[0])#注意reshape
    mask=np.isnan(col_fill)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = col_fill[np.arange(idx.shape[0])[:,None], idx]#向后补
    mask1=np.isnan(out)
    try:
        refer_value=out[0,np.argwhere(mask1[0]==False)[0][0]]#找到向后填充完成后第一个非nan值，用于填充前面的nan
        out[mask1]=refer_value #填充前面的nan
    except:
        return np.array(list(range(out.shape[1]))) * 0 - 1
    out=out.reshape(out.shape[1]) #注意再次reshape
    return out

def feature_fill(data):
    '''
        input: dataframe
        output: filled numpy
    '''
    data_np = np.array(data)
    for index in range(len(data_np[0])):
        data_np[:,index] = feature_fill_helper(data_np, index)
    return data_np

def feature_engineer(train):
    '''
        create new features        
        up-to-date feature we will use
        08/06 - kw - add some sofa features
    '''
    train = feature_fill(train)
    train = feature_engineer_cut(train)
    train[:,39] -= train[:,38]
    return train.copy()

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)

def fix_100(res, org_length):
    """
    input: 
            res label are given by predict
            org_length is the original length of each patient
            t is threshold
    output:
            fixed probabilities, original labels and predictions of original length,
            also processed by threshold
            
    """
    l = 100 # fixed length of each patient
    
    tmp_res = list(res)
    if org_length <= l:
        tmp_res = tmp_res[:org_length]
    else:
        last = tmp_res[-1]
        tmp_res += [last for _ in range(org_length - l)]

    tmp_predict = [1 for _ in range(len(tmp_res))]

    for r in range(len(tmp_res)):
        if tmp_res[r] < LSTM_t:
            tmp_predict[r] = 0
        else:
            tmp_res_ = tmp_res[:r] + [tmp_res[r] for _ in range(len(tmp_res) - r)]  
            return tmp_res_, tmp_predict

    return tmp_res, tmp_predict

def back_fix(predict):
    index_1 = np.where(predict >= LSTM_t)

    if len(index_1[0])>3:
        fix = index_1[0][3] + 1
        if fix >= len(predict):
            return predict
        predict[fix:] = np.ones((len(predict)-fix))
    return predict

def load_sepsis_model():
    
    #model = load_model(model_name) # you need this without self defined loss function
    model = load_model(model_name, custom_objects={'fmeasure':fmeasure}) # only for self-defined weighted binary crossentropy situation
    return model

def get_sepsis_score(data, model):
    
    LSTM_model = model
    
    # set up data
    # cur_train = pd.DataFrame(data, columns = org_feature) 
    cur_train = data   
    org_length = len(data)

    processed_train = feature_engineer(cur_train)
    cur_train = processed_train[:,feature_index]
    # xgb preclassifer
    xgb_flag = test(pd.DataFrame(processed_train, columns = org_feature)[:3].copy())

    # set up dtest
    dtest = cur_train
    dtest = dtest.reshape(-1,100,len(custom))
    preds = LSTM_model.predict(dtest)[0]
    
    #t = org_length if org_length < 100 else 99
    # xiuzheng
    # front_preds = []
    # if t >= 1:
    #     front_preds.append(preds[t-1])
    # if t >= 2:
    #     front_preds.append(preds[t-2])
    # if t >= 3:
    #     front_preds.append(preds[t-3])
    
    # tmp_fp = []
    # tmp_p = []
    # org_pred = preds[t]

    # for fp in front_preds:
    #     if fp >= threshold:
    #         tmp_fp.append(1)
    #         tmp_p.append(fp)
    #     elif fp < threshold:
    #         tmp_fp.append(0)
    #         tmp_p.append(1-fp)
    # tmp = 0
    # for i in range(len(tmp_p)):
    #     tmp += (tmp_p[i] * tmp_fp[i])
    
    # if len(tmp_p) >= 2:
    #     tmp /= len(tmp_p)
    #     preds[t] = org_pred * 0.9 + 0.1 * tmp
    # else:
    #     preds[t] = org_pred
    if org_length <12:
        if not xgb_flag:
            preds[:11] = [0 for _ in range(11)]
        if xgb_flag:
            preds[:3] = [0.7 for _ in range(3)]
    
    if org_length > 10:
        # print(predicted)
        preds= back_fix(preds)

    preds = [p if p > 0 else 0 for p in preds]
    preds = [p if p <= 1 else 1 for p in preds]

    #score, label = fix_100(preds, org_length)
    t = org_length if org_length < 100 else 99
    score = preds[t]
    label = 1 if score > LSTM_t else 0

    return score, label



