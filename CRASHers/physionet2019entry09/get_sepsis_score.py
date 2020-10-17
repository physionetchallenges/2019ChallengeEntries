#!/usr/bin/env python

import numpy as np
import xgboost as xgb

import sequence_processing as seqprc

def get_sepsis_score(data, model):
    
    # data header
    feature_labels = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', \
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 
    'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 
    'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

    # prepare input data
    X_ = seqprc.prepare_data(data, feature_labels, model['features'])
    X = np.c_[X_, seqprc.calculate_news(X_)]
    
    # get scores
    scores = model['model'].predict(xgb.DMatrix(X[-1:, :]))
    labels = (scores > model['thres'])
    
    return (scores, labels)


def load_sepsis_model():
    # load prediction model
    model_pred, model_pred_features = load_model_params('xgb_tr01_f03_mean')
    
    # load treshold
    thres = load_model_threshold('xgb_tr01_f03_mean_threshold.txt')
        
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
