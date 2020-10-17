import os
import re
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import math


def get_sepsis_score(data, model):
    
    def expiry_data(data):

        data['LabTest'] = len(lab_cols)- data[lab_cols].isnull().sum(axis = 1)
        ## Unit:
        data[['Unit1', 'Unit2']] = data[['Unit1', 'Unit2']].fillna(0)
        data['Unit2'][data['Unit2'] == 1] = 2
        data['Unit'] = data[['Unit1', 'Unit2']].max(axis = 1)
        data = data.drop(['Unit1', 'Unit2'], axis = 1)

        df = data[HR_cols_all]
        df = df.applymap(lambda x: x if math.isnan(x) else 2 )
        df.columns = HR_cols_all_NW
        data = pd.concat([data,df], axis = 1)
        ## Fill Forward
        data[HR_cols_1_NW] = data[HR_cols_1_NW].fillna(method='ffill', limit = 1)
        data[HR_cols_4_NW] = data[HR_cols_4_NW].fillna(method='ffill', limit = 4)
        data[HR_cols_6_NW] = data[HR_cols_6_NW].fillna(method='ffill', limit = 6)
        data[HR_cols_14_NW] = data[HR_cols_14_NW].fillna(method='ffill', limit = 14)

        ## Fill valid period:
        data.loc[(data.HR_NW == 2) & ((data.HR).isnull()), 'HR_NW'] = 1
        data.loc[(data.O2Sat_NW == 2) & ((data.O2Sat).isnull()), 'O2Sat_NW'] = 1
        data.loc[(data.SBP_NW == 2) & ((data.SBP).isnull()), 'SBP_NW'] = 1
        data.loc[(data.MAP_NW == 2) & ((data.MAP).isnull()), 'MAP_NW'] = 1
        data.loc[(data.DBP_NW == 2) & ((data.DBP).isnull()), 'DBP_NW'] = 1
        data.loc[(data.Resp_NW == 2) & ((data.Resp).isnull()), 'Resp_NW'] = 1
        data.loc[(data.EtCO2_NW == 2) & ((data.EtCO2).isnull()), 'EtCO2_NW'] = 1

        data.loc[(data.Temp_NW == 2) & ((data.Temp).isnull()), 'Temp_NW'] = 1
        data.loc[(data.FiO2_NW == 2) & ((data.FiO2).isnull()), 'FiO2_NW'] = 1
        data.loc[(data.pH_NW == 2) & ((data.pH).isnull()), 'pH_NW'] = 1
        data.loc[(data.PaCO2_NW == 2) & ((data.PaCO2).isnull()), 'PaCO2_NW'] = 1
        data.loc[(data.SaO2_NW == 2) & ((data.SaO2).isnull()), 'SaO2_NW'] = 1
        data.loc[(data.BaseExcess_NW == 2) & ((data.BaseExcess).isnull()), 'BaseExcess_NW'] = 1
        data.loc[(data.HCO3_NW == 2) & ((data.HCO3).isnull()), 'HCO3_NW'] = 1

        data.loc[(data.AST_NW == 2) & ((data.AST).isnull()), 'AST_NW'] = 1
        data.loc[(data.BUN_NW == 2) & ((data.BUN).isnull()), 'BUN_NW'] = 1
        data.loc[(data.Glucose_NW == 2) & ((data.Glucose).isnull()), 'Glucose_NW'] = 1

        data.loc[(data.Bilirubin_direct_NW == 2) & ((data.Bilirubin_direct).isnull()), 'Bilirubin_direct_NW'] = 1
        data.loc[(data.Lactate_NW == 2) & ((data.Lactate).isnull()), 'Lactate_NW'] = 1
        data.loc[(data.Magnesium_NW == 2) & ((data.Magnesium).isnull()), 'Magnesium_NW'] = 1
        data.loc[(data.Phosphate_NW == 2) & ((data.Phosphate).isnull()), 'Phosphate_NW'] = 1
        data.loc[(data.Potassium_NW == 2) & ((data.Potassium).isnull()), 'Potassium_NW'] = 1
        data.loc[(data.Bilirubin_total_NW == 2) & ((data.Bilirubin_total).isnull()), 'Bilirubin_total_NW'] = 1
        data.loc[(data.TroponinI_NW == 2) & ((data.TroponinI).isnull()), 'TroponinI_NW'] = 1
        data.loc[(data.Platelets_NW == 2) & ((data.Platelets).isnull()), 'Platelets_NW'] = 1
        data.loc[(data.Hct_NW == 2) & ((data.Hct).isnull()), 'Hct_NW'] = 1
        data.loc[(data.Hgb_NW == 2) & ((data.Hgb).isnull()), 'Hgb_NW'] = 1
        data.loc[(data.PTT_NW == 2) & ((data.PTT).isnull()), 'PTT_NW'] = 1
        data.loc[(data.WBC_NW == 2) & ((data.WBC).isnull()), 'WBC_NW'] = 1
        data.loc[(data.Fibrinogen_NW == 2) & ((data.Fibrinogen).isnull()), 'Fibrinogen_NW'] = 1


        ## At the end anything left in _NW is 0:
        data[HR_cols_all_NW] = data[HR_cols_all_NW].fillna(0)

        ## Actuall Values:
        data = data.fillna(method = 'ffill')
        data = data.fillna(Normal_value_dic)  
    
        return data
    
    org_col_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
       'HospAdmTime', 'ICULOS']
    Normal_value_dic = {'HR': 75 , 'O2Sat': 100, 'Temp': 37, 'SBP': 120, 'MAP': 85, 'DBP': 80, 'Resp': 14, 'EtCO2': 40,
           'BaseExcess': 0, 'HCO3': 25.5, 'FiO2': 0.500, 'pH':7.4, 'PaCO2':85, 'SaO2': 100, 'AST': 37.5, 'BUN':13.5,
           'Alkalinephos': 90, 'Calcium': 9.35, 'Chloride': 101 , 'Creatinine':0.9, 'Bilirubin_direct':0.3,
           'Glucose':90, 'Lactate':2.2, 'Magnesium':2, 'Phosphate':3.5, 'Potassium':4.25,
           'Bilirubin_total': 0.75, 'TroponinI':0.04, 'Hct':48.5, 'Hgb':13.75, 'PTT':30, 'WBC':7.7,
           'Fibrinogen':375, 'Platelets':300}
    lab_cols = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
           'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
           'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
           'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
           'Fibrinogen', 'Platelets']
    feature_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
               'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
               'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
               'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
               'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
               'Fibrinogen', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS',
                'LabTest']
    NW_featurs = ['HR_NW', 'O2Sat_NW', 'SBP_NW',
               'MAP_NW', 'DBP_NW', 'Resp_NW', 'EtCO2_NW', 'Temp_NW', 'FiO2_NW',
               'pH_NW', 'PaCO2_NW', 'SaO2_NW', 'BaseExcess_NW', 'HCO3_NW', 'AST_NW',
               'BUN_NW', 'Glucose_NW', 'Bilirubin_direct_NW', 'Lactate_NW',
               'Magnesium_NW', 'Phosphate_NW', 'Potassium_NW', 'Bilirubin_total_NW',
               'TroponinI_NW', 'Hct_NW', 'Hgb_NW', 'PTT_NW', 'WBC_NW', 'Fibrinogen_NW',
               'Platelets_NW', 'Unit']
    features = feature_names + NW_featurs

    HR_cols_1 = ['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    HR_cols_1_NW = [x + '_NW' for x in HR_cols_1]

    HR_cols_4 = ['Temp','FiO2', 'pH', 'PaCO2', 'SaO2','BaseExcess', 'HCO3', 'AST', 'BUN']
    HR_cols_4_NW = [x + '_NW' for x in HR_cols_4]

    HR_cols_6 = ['Glucose']
    HR_cols_6_NW = [x + '_NW' for x in HR_cols_6]

    HR_cols_14 = ['Bilirubin_direct','Lactate', 'Magnesium', 'Phosphate', 'Potassium',
                  'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
                  'Fibrinogen', 'Platelets']
    HR_cols_14_NW = [x + '_NW' for x in HR_cols_14]

    HR_cols_all = HR_cols_1 + HR_cols_4 + HR_cols_6 + HR_cols_14
    HR_cols_all_NW = [x + '_NW' for x in HR_cols_all]

    threshold = 0.02
    
    data = pd.DataFrame(data, columns = org_col_names)
    data = expiry_data(data)
    data = data[features] 
    y_pred = model.predict_proba(data)
    scores = y_pred[:,1]
    labels = (scores >= threshold).astype('int')
    return scores[-1], labels[-1]


def load_sepsis_model():
    xgboost_path = "XGBoost_All_FeatExtract_Expiry.txt"
    model = pickle.load(open(xgboost_path, 'rb'))
    return model
    