#!/usr/bin/env python

import numpy as np
import pandas as pd

import sequence_processing as seqprc

import xgboost as xgb

def imputeMissingData(X):
    
    # average for all the features included in the model
    x_mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777,
        60.8711, 1.0, 0.0, 0.0, -59.6769,
        0.68024, 2.8557, 
        0.0, 0.0, 0.0, 0.0,
        83.8996, 36.8055, 126.2240, 66.2070, 86.2907, 18.7280,
        83.8996, 36.8055, 126.2240, 66.2070, 86.2907, 18.7280, 97.0520, 
        83.8996, 36.8055, 126.2240, 66.2070, 86.2907, 18.7280,
        83.8996, 36.8055, 126.2240, 66.2070, 86.2907, 18.7280, 97.0520])
    
    #Find indicies that you need to replace
    inds = np.where(np.isnan(X))
    
    #Place column means in the indices. Align the arrays using take
    X[inds] = np.take(x_mean, inds[1])
    
    return X
    
def get_sepsis_score(data, model):

    # create a data frame
    feature_labels = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', \
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 
    'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 
    'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

    df = pd.DataFrame(data=data, columns=feature_labels)
    
    # clean data
    df = seqprc.clean_vitals(df)
    # compute missing flag data (which includes timing since observation)
    df = seqprc.computeMissingFlagData(df,feature_labels[:34])
    # calculate change in value for certain labs
    df1, _ = seqprc.augmentChangeOverWindow(df, ['Creatinine', 'BUN', 'Bilirubin_total', 'Platelets'])
    # need to fill missing data
    df1 = seqprc.fillMissingData(df1, features=feature_labels[:34])
    # augment data
    df1, _ = seqprc.augmentMaximumOverWindow(df1, ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp'], WIN=12)
    df1, _ = seqprc.augmentMinimumOverWindow(df1, ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp', 'O2Sat'], WIN=12)
    df1, _ = seqprc.augmentMaximumOverWindow(df1, ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp'], WIN=24)
    df1, _ = seqprc.augmentMinimumOverWindow(df1, ['HR', 'Temp', 'SBP', 'DBP', 'MAP', 'Resp', 'O2Sat'], WIN=24)
    
    df1, _ = seqprc.augmentInteractions(df1)
    
    ### set model params
    feature_map = model['features']
    model_score = model['model'] 
    model_thres = model['thres']
    
    # convert data to xgb entry
    dtest = imputeMissingData(df1[feature_map].values)
    dtest_xgb = xgb.DMatrix(dtest)

    # get scores
    scores = model_score.predict(dtest_xgb)
    labels = (scores > model_thres)
    
    return (scores, labels)


def load_sepsis_model():
    # load prediction model
    model_pred, model_pred_features = load_model_params('xgb_tr01_f01_mean')
    
    # load treshold
    thres = load_model_threshold('xgb_tr01_f01_mean_threshold.txt')
        
    full_model = {
        'model': model_pred, 
        'features': model_pred_features, 
        'thres': thres 
    }
    
    return full_model


def load_model_params(model_name):
    model = xgb.Booster()
    model.load_model(model_name)
    
    model_features = []
    with open(model_name + '_features.txt', 'r') as filehandle:
        for line in filehandle:
            # remove line break
            current_feature = line[:-1]
            # add feature to feature list
            model_features.append(current_feature)
    
    return model, model_features


def load_model_threshold(filename):
    with open(filename, 'r') as filehandle:
        linelist = filehandle.readlines()
        thres = linelist[0]
    
    return float(thres)