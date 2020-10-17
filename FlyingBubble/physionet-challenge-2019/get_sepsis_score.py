#!/usr/bin/env python

import numpy as np
import pandas as pd
import lightgbm as lgb
import time
feature_list = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 
       'HospAdmTime', 'ICULOS', 'WBC_24h_max',
       'WBC_24h_min', 'Temp_24h_max', 'Temp_24h_min', 'MAP_24h_max',
       'MAP_24h_min', 'SBP_24h_max',
    'SBP_24h_min', 'Creatinine_24h_max', 'Creatinine_24h_min',
       'Platelets_24h_max', 'Platelets_24h_min', 'FiO2_24h_max',
       'FiO2_24h_min', 'SaO2_24h_max', 'SaO2_24h_min', 'PTT_24h_max',
       'PTT_24h_min', 'BUN_24h_max', 'BUN_24h_min', 'Calcium_24h_max',
       'Calcium_24h_min', 'Phosphate_24h_max', 'Phosphate_24h_min',
       'Hct_24h_max', 'Hct_24h_min', 'Lactate_24h_max', 'Lactate_24h_min',
       'Alkalinephos_24h_max', 'Alkalinephos_24h_min', 'Glucose_24h_max',
       'Glucose_24h_min', 'Hgb_24h_max', 'Hgb_24h_min', 'WBC_12h_max',
       'WBC_12h_min', 'Temp_12h_max', 'Temp_12h_min', 'MAP_12h_max',
       'MAP_12h_min', 'SBP_12h_max', 'SBP_12h_min', 'Creatinine_12h_max',
       'Creatinine_12h_min', 'Platelets_12h_max', 'Platelets_12h_min',
       'FiO2_12h_max', 'FiO2_12h_min', 'SaO2_12h_max', 'SaO2_12h_min',
       'PTT_12h_max', 'PTT_12h_min', 'BUN_12h_max', 'BUN_12h_min',
       'Calcium_12h_max', 'Calcium_12h_min', 'Phosphate_12h_max',
    'Phosphate_12h_min', 'Hct_12h_max', 'Hct_12h_min', 'Lactate_12h_max',
       'Lactate_12h_min', 'Alkalinephos_12h_max', 'Alkalinephos_12h_min',
       'Glucose_12h_max', 'Glucose_12h_min', 'Hgb_12h_max', 'Hgb_12h_min',
       'WBC_6h_max', 'WBC_6h_min', 'Temp_6h_max', 'Temp_6h_min', 'MAP_6h_max',
       'MAP_6h_min', 'SBP_6h_max', 'SBP_6h_min', 'Creatinine_6h_max',
       'Creatinine_6h_min', 'Platelets_6h_max', 'Platelets_6h_min',
       'FiO2_6h_max', 'FiO2_6h_min', 'SaO2_6h_max', 'SaO2_6h_min',
       'PTT_6h_max', 'PTT_6h_min', 'BUN_6h_max', 'BUN_6h_min',
       'Calcium_6h_max', 'Calcium_6h_min', 'Phosphate_6h_max',
       'Phosphate_6h_min', 'Hct_6h_max', 'Hct_6h_min', 'Lactate_6h_max',
       'Lactate_6h_min', 'Alkalinephos_6h_max', 'Alkalinephos_6h_min',
       'Glucose_6h_max', 'Glucose_6h_min', 'Hgb_6h_max', 'Hgb_6h_min',
       'HR_div_SBP', 'SaO2_div_FiO2', 'counts_24h', 'Inspection_frequency',
       'WBC_24h_diff',
    'Temp_24h_diff', 'MAP_24h_diff', 'SBP_24h_diff', 'Creatinine_24h_diff',
       'Platelets_24h_diff', 'FiO2_24h_diff', 'SaO2_24h_diff', 'PTT_24h_diff',
       'BUN_24h_diff', 'Calcium_24h_diff', 'Phosphate_24h_diff',
       'Hct_24h_diff', 'Lactate_24h_diff', 'Alkalinephos_24h_diff',
       'Glucose_24h_diff', 'Hgb_24h_diff', 'ICUAdmTime', 'HR_before_one',
       'HR_before_two', 'MAP_before_one', 'MAP_before_two', 'O2Sat_before_one',
       'O2Sat_before_two', 'Resp_before_one', 'Resp_before_two',
       'SBP_before_one', 'SBP_before_two', 'HR_diff_two',
       'MAP_diff_two', 'O2Sat_diff_two',
       'Resp_diff_two', 'SBP_diff_one', 'SBP_diff_two',
       'Temp_before_one', 'Temp_before_two', 'Temp_diff_one', 'Temp_diff_two',
       'HR_slope_one', 'HR_slope_two', 'MAP_slope_one', 'MAP_slope_two',
       'O2Sat_slope_one', 'O2Sat_slope_two', 'Resp_slope_one',
       'Resp_slope_two', 'SBP_slope_one', 'SBP_slope_two', 'Temp_slope_one',
       'Temp_slope_two', 'WBC_12h_diff', 'Temp_12h_diff', 'MAP_12h_diff',
       'SBP_12h_diff', 'Creatinine_12h_diff', 'Platelets_12h_diff',
       'FiO2_12h_diff', 'SaO2_12h_diff', 'PTT_12h_diff', 'BUN_12h_diff',
       'Calcium_12h_diff', 'Phosphate_12h_diff', 'Hct_12h_diff',
       'Lactate_12h_diff', 'Alkalinephos_12h_diff', 'Glucose_12h_diff',
       'Hgb_12h_diff', 'WBC_6h_diff', 'Temp_6h_diff', 'MAP_6h_diff',
       'SBP_6h_diff', 'Creatinine_6h_diff', 'Platelets_6h_diff',
       'FiO2_6h_diff', 'SaO2_6h_diff', 'PTT_6h_diff', 'BUN_6h_diff',
       'Calcium_6h_diff', 'Phosphate_6h_diff', 'Hct_6h_diff',
       'Lactate_6h_diff', 'Alkalinephos_6h_diff', 'Glucose_6h_diff',
       'Hgb_6h_diff', 'qSOFA', 'Glucose_before_one', 'Glucose_before_two',
       'WBC_before_one', 'WBC_before_two',
       'Glucose_diff_two', 'WBC_diff_two', 'WBC_slope_two'
]

columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
       'HospAdmTime', 'ICULOS']

max_min_list = ['WBC','Temp','MAP','SBP','Creatinine','Platelets','FiO2','SaO2',
               'PTT','BUN','Calcium','Phosphate','Hct','Lactate','Alkalinephos','Glucose','Hgb']

maxmin_cols = ['WBC_24h_max', 'Temp_24h_max', 'MAP_24h_max', 'SBP_24h_max', 'Creatinine_24h_max', 'Platelets_24h_max',
               'FiO2_24h_max', 'SaO2_24h_max', 'PTT_24h_max', 'BUN_24h_max', 'Calcium_24h_max', 'Phosphate_24h_max',
               'Hct_24h_max', 'Lactate_24h_max', 'Alkalinephos_24h_max', 'Glucose_24h_max', 'Hgb_24h_max',
               'WBC_12h_max', 'Temp_12h_max', 'MAP_12h_max', 'SBP_12h_max', 'Creatinine_12h_max', 'Platelets_12h_max',
               'FiO2_12h_max', 'SaO2_12h_max', 'PTT_12h_max', 'BUN_12h_max', 'Calcium_12h_max', 'Phosphate_12h_max',
               'Hct_12h_max', 'Lactate_12h_max', 'Alkalinephos_12h_max', 'Glucose_12h_max', 'Hgb_12h_max', 'WBC_6h_max',
               'Temp_6h_max', 'MAP_6h_max', 'SBP_6h_max', 'Creatinine_6h_max', 'Platelets_6h_max', 'FiO2_6h_max',
               'SaO2_6h_max', 'PTT_6h_max', 'BUN_6h_max', 'Calcium_6h_max', 'Phosphate_6h_max', 'Hct_6h_max',
               'Lactate_6h_max', 'Alkalinephos_6h_max', 'Glucose_6h_max', 'Hgb_6h_max', 'WBC_24h_min', 'Temp_24h_min',
               'MAP_24h_min', 'SBP_24h_min', 'Creatinine_24h_min', 'Platelets_24h_min', 'FiO2_24h_min', 'SaO2_24h_min',
               'PTT_24h_min', 'BUN_24h_min', 'Calcium_24h_min', 'Phosphate_24h_min', 'Hct_24h_min', 'Lactate_24h_min',
               'Alkalinephos_24h_min', 'Glucose_24h_min', 'Hgb_24h_min', 'WBC_12h_min', 'Temp_12h_min', 'MAP_12h_min',
               'SBP_12h_min', 'Creatinine_12h_min', 'Platelets_12h_min', 'FiO2_12h_min', 'SaO2_12h_min', 'PTT_12h_min',
               'BUN_12h_min', 'Calcium_12h_min', 'Phosphate_12h_min', 'Hct_12h_min', 'Lactate_12h_min',
               'Alkalinephos_12h_min', 'Glucose_12h_min', 'Hgb_12h_min', 'WBC_6h_min', 'Temp_6h_min', 'MAP_6h_min',
               'SBP_6h_min', 'Creatinine_6h_min', 'Platelets_6h_min', 'FiO2_6h_min', 'SaO2_6h_min', 'PTT_6h_min',
               'BUN_6h_min', 'Calcium_6h_min', 'Phosphate_6h_min', 'Hct_6h_min', 'Lactate_6h_min',
               'Alkalinephos_6h_min', 'Glucose_6h_min', 'Hgb_6h_min', 'WBC_24h_diff', 'Temp_24h_diff', 'MAP_24h_diff',
               'SBP_24h_diff', 'Creatinine_24h_diff', 'Platelets_24h_diff', 'FiO2_24h_diff', 'SaO2_24h_diff',
               'PTT_24h_diff', 'BUN_24h_diff', 'Calcium_24h_diff', 'Phosphate_24h_diff', 'Hct_24h_diff',
               'Lactate_24h_diff', 'Alkalinephos_24h_diff', 'Glucose_24h_diff', 'Hgb_24h_diff', 'WBC_12h_diff',
               'Temp_12h_diff', 'MAP_12h_diff', 'SBP_12h_diff', 'Creatinine_12h_diff', 'Platelets_12h_diff',
               'FiO2_12h_diff', 'SaO2_12h_diff', 'PTT_12h_diff', 'BUN_12h_diff', 'Calcium_12h_diff',
               'Phosphate_12h_diff', 'Hct_12h_diff', 'Lactate_12h_diff', 'Alkalinephos_12h_diff', 'Glucose_12h_diff',
               'Hgb_12h_diff', 'WBC_6h_diff', 'Temp_6h_diff', 'MAP_6h_diff', 'SBP_6h_diff', 'Creatinine_6h_diff',
               'Platelets_6h_diff', 'FiO2_6h_diff', 'SaO2_6h_diff', 'PTT_6h_diff', 'BUN_6h_diff', 'Calcium_6h_diff',
               'Phosphate_6h_diff', 'Hct_6h_diff', 'Lactate_6h_diff', 'Alkalinephos_6h_diff', 'Glucose_6h_diff',
               'Hgb_6h_diff']

sds_cols = ['HR_before_one','MAP_before_one','O2Sat_before_one','Resp_before_one',
 'SBP_before_one','Temp_before_one','Glucose_before_one','WBC_before_one',
 'HR_before_two','MAP_before_two','O2Sat_before_two','Resp_before_two',
 'SBP_before_two','Temp_before_two','Glucose_before_two','WBC_before_two',
 'HR_diff_one','MAP_diff_one','O2Sat_diff_one','Resp_diff_one','SBP_diff_one',
 'Temp_diff_one','Glucose_diff_one','WBC_diff_one',
 'HR_diff_two','MAP_diff_two','O2Sat_diff_two','Resp_diff_two','SBP_diff_two',
 'Temp_diff_two','Glucose_diff_two','WBC_diff_two',
 'HR_slope_one','MAP_slope_one','O2Sat_slope_one','Resp_slope_one',
 'SBP_slope_one','Temp_slope_one','Glucose_slope_one','WBC_slope_one',
 'HR_slope_two','MAP_slope_two','O2Sat_slope_two','Resp_slope_two',
 'SBP_slope_two','Temp_slope_two','Glucose_slope_two','WBC_slope_two'
]

# 处理异常值
def O2Sat_level(x):
    if x<70:
        return 0
    elif x<90:
        return 1
    else:
        return 2

def Temp_x(x):
    if x>41.6:
        return 41.6
    elif x<32.36:
        return 32.36
    else:
        return x

def FiO2_x(x):
    if x<0.21:
        return 0.21
    elif x>1:
        return 1
    else:
        return x
    
def get_sepsis_score(data, model):
    data = pd.DataFrame(data,columns=columns)
    
    # 统计检查次数
    
    counts_24h = data[-24:][['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets','ICULOS']].count().sum()
    data['counts_24h'] = counts_24h
    data['Inspection_frequency'] = data['counts_24h'] / data['ICULOS'].map(lambda x:x if x<24 else 24)
#     data.apply(lambda df:df['counts_24h']/df['ICULOS'] if df['ICULOS']<24 else df['counts_24h']/24,axis=1)
    # 异常值处理
    
    data['O2Sat_level'] = data['O2Sat'].map(O2Sat_level)
    data['Temp'] = data['Temp'].map(Temp_x)
    data['FiO2'] = data['FiO2'].map(FiO2_x)
    
    # 前向插值 
    
    data = data.fillna(method='ffill')[-24:]
    
    
    # 特征构造
    maxmin_arr = data[max_min_list].values
    
    max24h = np.nanmax(maxmin_arr[-24:,:], axis=0).reshape(1,-1)
    max12h = np.nanmax(maxmin_arr[-12:,:], axis=0).reshape(1,-1)    
    max6h = np.nanmax(maxmin_arr[-6:,:], axis=0).reshape(1,-1)
    min24h = np.nanmin(maxmin_arr[-24:,:],axis=0).reshape(1,-1)
    min12h = np.nanmin(maxmin_arr[-12:,:],axis=0).reshape(1,-1)
    min6h = np.nanmin(maxmin_arr[-6:,:],axis=0).reshape(1,-1)
    diff24h = max24h-min24h
    diff12h = max12h-min12h
    diff6h = max6h-min6h   
    maxmin_arr2 = np.concatenate((max24h,max12h,max6h,min24h,min12h,min6h,diff24h,diff12h,diff6h), axis=1)
    maxmin_arr3 = np.tile(maxmin_arr2, (data.shape[0], 1))
    maxmin_df = pd.DataFrame(maxmin_arr3, columns=maxmin_cols)
    
    
    data = pd.concat([data.reset_index(drop=True), maxmin_df], axis=1)

    
    data['HR_div_SBP'] = data['HR'] / data['SBP']
    data['SaO2_div_FiO2'] = data['SaO2'] / data['FiO2']
    data['ICUAdmTime'] = data['HospAdmTime'] - data['ICULOS']
    
    
    data['qSOFA'] = data['SBP'].map(lambda x:1 if x<=100 else 0) +data['Resp'].map(lambda x:1 if x>=22 else 0)
    
    
    shift0 = data[['HR','MAP','O2Sat','Resp','SBP','Temp','Glucose','WBC']]
    shift1 = shift0.shift(1)
    shift2 = shift0.shift(2)
    
    diff_one = shift0 - shift1
    diff_two = shift0 - shift2
    
    slope_one = diff_one / shift1
    slope_two = diff_two / shift2
    
    sds = pd.DataFrame(np.array(pd.concat([shift1,shift2,diff_one,diff_two,slope_one,slope_two], axis=1)),
                      columns = sds_cols)
    data = pd.concat([data,sds], axis=1)
    
    
#     for col in ['HR','MAP','O2Sat','Resp','SBP','Temp','Glucose','WBC']:
        
#         data[col+'_before_one'] = data[col].shift(1)             
#         data[col+'_before_two'] = data[col].shift(2)
        
        
#         data[col+'_diff_one'] = data[col] - data[col+'_before_one']
#         data[col+'_diff_two'] = data[col] - data[col+'_before_two']
        
        
#         data[col+'_slope_one'] = data[col+'_diff_one'] / data[col+'_before_one']
#         data[col+'_slope_two'] = data[col+'_diff_two'] / data[col+'_before_two']
    
        
    score = (0.3*model[0].predict(np.array(data[feature_list][-1:]))[0]+0.1*model[1].predict(np.array(data[feature_list][-1:]))[0]+0.15*model[2].predict(np.array(data[feature_list][-1:]))[0]+0.15*model[3].predict(np.array(data[feature_list][-1:]))[0]+0.3*model[4].predict(np.array(data[feature_list][-1:]))[0])
    label = score>0.023
    return score, label

def load_sepsis_model():
    
    model_0 = lgb.Booster(model_file='./lightGBM_237features_8398_0.txt')
    model_1 = lgb.Booster(model_file='./lightGBM_237features_8398_1.txt')
    model_2 = lgb.Booster(model_file='./lightGBM_237features_8398_2.txt')
    model_3 = lgb.Booster(model_file='./lightGBM_237features_8398_3.txt')
    model_4 = lgb.Booster(model_file='./lightGBM_237features_8398_4.txt')
    model_list = [model_0,model_1,model_2,model_3,model_4]
    return model_list

