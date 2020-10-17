#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def get_sepsis_score(data, model):

    
    col = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
       'HospAdmTime', 'ICULOS']
    
    data = pd.DataFrame(data, columns = col)

    data.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos','Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' , 'Resp','Lactate','Magnesium','Phosphate','Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',"Unit1", 'Unit2', "Glucose",'DBP','SBP'], axis = 1, inplace = True)
    data.fillna(method='bfill',inplace=True)
    data.fillna(method='ffill',inplace=True)
    

    
    if sum(pd.isna(data['MAP'])) !=0 : 
        data['MAP'] =  [87] * len(data['MAP'] )
    if sum(pd.isna(data['Temp'])) !=0 : 
        data['Temp'] =  [36.7] * len(data['Temp'] )
    if sum(pd.isna(data['HR'])) !=0 : 
        data['HR'] =  [83.9] * len(data['HR'] )
    if sum(pd.isna(data['O2Sat'])) !=0 : 
        data['O2Sat'] =  [95] * len(data['O2Sat'] )
    
    probabilities =  model.predict_proba(data)[:,1]
    
    for i in range (len(probabilities)):
        if probabilities[i] < 0.55 :   
            predictions = 0
        else :
            predictions = 1


    score = probabilities[0]

    return score, predictions



def load_sepsis_model():
    loaded_model = joblib.load('regression_model.sav')
    return(loaded_model)







