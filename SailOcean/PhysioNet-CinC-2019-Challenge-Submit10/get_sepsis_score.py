#!/usr/bin/env python

import numpy as np
import pandas as pd
import xgboost as xgb

def feature_extract(data, sep_index):
    temp = data.copy()   
    last_raw = temp[-1]
    for sep_column in sep_index:
        sep_data = temp[:, sep_column]
        nan_pos = np.where(~np.isnan(sep_data))[0]
        if len(nan_pos) == 0:
            interval_f1 =  0
            last_raw = np.append(last_raw, interval_f1)
            interval_f2 = -1
            last_raw = np.append(last_raw, interval_f2)
        else:
            interval_f1 = len(nan_pos)
            last_raw = np.append(last_raw, interval_f1)
            interval_f2 = len(temp) -1 - nan_pos[-1]
            last_raw = np.append(last_raw, interval_f2)

        diff_data = sep_data
        if len(nan_pos) <= 1:
            diff_f = np.NaN
            last_raw = np.append(last_raw, diff_f)
        else:
            diff_f = diff_data[nan_pos[-1]] - diff_data[nan_pos[-2]]
            last_raw = np.append(last_raw, diff_f)

        last_raw_interval = last_raw[40:]
    
    return last_raw_interval

def features_score(data):
    dat = data[-1]
    news_score = np.zeros((1, 8))

    HR = dat[0]
    if HR == np.nan:
        HR_score = np.nan
    elif (HR <= 40) | (HR >= 131):
        HR_score = 3
    elif 111 <= HR <= 130:
        HR_score = 2
    elif (41 <= HR <= 50) | (91 <= HR <= 110):
        HR_score = 1
    else:
        HR_score = 0
    news_score[0, 0] = HR_score

    Temp = dat[2]
    if Temp == np.nan:
        Temp_score = np.nan
    elif Temp <= 35:
        Temp_score = 3
    elif Temp >= 39.1:
        Temp_score = 2
    elif (35.1 <= Temp <= 36.0) | (38.1 <= Temp <= 39.0):
        Temp_score = 1
    else:
        Temp_score = 0
    news_score[0, 1] = Temp_score

    Resp = dat[6]
    if Resp == np.nan:
        Resp_score = np.nan
    elif (Resp < 8) | (Resp > 25):
        Resp_score = 3
    elif 21 <= Resp <= 24:
        Resp_score = 2
    elif 9 <= Resp <= 11:
        Resp_score = 1
    else:
        Resp_score = 0
    news_score[0, 2] = Resp_score

    Creatinine = dat[19]
    if Creatinine == np.nan:
        Creatinine_score = np.nan
    elif Creatinine < 1.2:
        Creatinine_score = 0
    elif Creatinine < 2:
        Creatinine_score = 1
    elif Creatinine < 3.5:
        Creatinine_score = 2
    else:
        Creatinine_score = 3
    news_score[0, 3] = Creatinine_score

    MAP = dat[4]
    if MAP == np.nan:
        MAP_score = np.nan
    elif MAP >= 70:
        MAP_score = 0
    else:
        MAP_score = 1
    news_score[0, 4] = MAP_score

    SBP = dat[3]
    if SBP + Resp == np.nan:
        qsofa = np.nan
    elif (SBP <= 100) & (Resp >= 22):
        qsofa = 1
    else:
        qsofa = 0
    news_score[0, 5] = qsofa

    Platelets = dat[33]
    if Platelets == np.nan:
        Platelets_score = np.nan
    elif Platelets <= 50:
        Platelets_score = 3
    elif Platelets <= 100:
        Platelets_score = 2
    elif Platelets <= 150:
        Platelets_score = 1
    else:
        Platelets_score = 0
    news_score[0, 6] = Platelets_score

    Bilirubin = dat[26]
    if Bilirubin == np.nan:
        Bilirubin_score = np.nan
    elif Bilirubin < 1.2:
        Bilirubin_score = 0
    elif Bilirubin < 2:
        Bilirubin_score = 1
    elif Bilirubin < 6:
        Bilirubin_score = 2
    else:
        Bilirubin_score = 3
    news_score[0, 7] = Bilirubin_score

    news_score = news_score[-1]
    return news_score

def feature_window(data, win_index):
    sepdata = data[:, win_index].copy()
    features = np.zeros(shape=(6, len(win_index)))
    
    if len(sepdata) < 7:
        win_data = sepdata[0:len(sepdata)]
        for ii in range(6-(len(sepdata)-1)):
            win_data = np.row_stack((win_data, sepdata[-1]))
    else:
        win_data = sepdata[-7 : ]
        
    for j in range(len(win_index)):
        dat = win_data[:, j]
        if len(np.where(~np.isnan(dat))[0]) == 0:
            features[:,j] = np.nan
        else:
            features[0,j] = np.nanmax(dat)
            features[1,j] = np.nanmin(dat)
            features[2,j] = np.nanmean(dat)
            features[3,j] = np.nanmedian(dat)
            features[4,j] = np.nanstd(dat)
            features[5,j] = np.std(np.diff(dat))
    features = features.flatten()

    return features

def get_sepsis_score(data, model):
    index_all = np.arange(0, 40)
    feature_extract_index = np.delete(index_all, [20, 27, 32, 34,35,36,37,38,39])
    last_raw_interval = feature_extract(data, feature_extract_index)
    data = pd.DataFrame(data)
    data = data.fillna(method = 'ffill')
    data = np.array(data)

    fea_score = features_score(data)

    win_index = [0, 1, 3, 4, 6]
    features = feature_window(data, win_index)
    data = np.delete(data, [20, 27, 32], axis=1)
    data_last = (np.array(data))[-1]
    data_last = np.concatenate((data_last, last_raw_interval, features, fea_score))
    data_last = data_last.reshape((1,len(data_last)))

    X = xgb.DMatrix(data_last)

    predict_pro = np.zeros((1, 5))
    for k in range(5):
        predict_pro[0,k] = model[k].predict(X)

    score = np.mean(predict_pro)
    label = score > 0.522

    return score, label

def load_sepsis_model():
    xgb_model1 = xgb.Booster(model_file = 'model1.mdl')
    xgb_model2 = xgb.Booster(model_file = 'model2.mdl')
    xgb_model3 = xgb.Booster(model_file = 'model3.mdl')
    xgb_model4 = xgb.Booster(model_file = 'model4.mdl')
    xgb_model5 = xgb.Booster(model_file = 'model5.mdl')
    xgb_model = [xgb_model1, xgb_model2, xgb_model3, xgb_model4, xgb_model5]

    return xgb_model
