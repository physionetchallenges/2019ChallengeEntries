#!/usr/bin/env python

import sys
import numpy as np, pandas as pd 
import lightgbm as lgb
import os


def load_sepsis_model():
    
#     from keras.models import load_model '/Users/macbook/OneDrive - North Dakota University System/'
#     DIR= '/Users/macbook/OneDrive - North Dakota University System/PhysioNet Scripts/2019/Notebooks/LGBM/'
    model = lgb.Booster(model_file='lgb_classifier.txt')
    
    return model




def load_challenge_data(file):
    subject=  pd.read_csv(file, sep = "|")

# with timer("Create Time To Failure columns"):
    TTF= [f for f in subject.SepsisLabel.values if f==0 ]
    ttf_len= len(TTF)
    
    
    subject['Subject_ID']= file.split(".")[0].split("/")[-1]
    subject['ICULOS']= subject.reset_index()['index'].values
    subject['TTF']= [f for f in range(ttf_len, 0,-1)] + [0]*(len(subject) - ttf_len)
    
    
    
    
    data =subject.set_index(['Subject_ID', 'TTF']).copy()


    y_cols= ["TTF", 'SepsisLabel']
    features = [f for f in data.columns if f not in y_cols+ ['Subject_ID', 'ICULOS']]
    df= data.reset_index().copy()
 
    df= df[features].interpolate()
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    from sklearn import preprocessing
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, index=data.index, columns=features)
    
    
    
    dat1= data.reset_index()[['Subject_ID', 'ICULOS']+y_cols]
    
    df = pd.concat([dat1, df.reset_index()[features]], axis=1) 
            
    data = df[features].values
    
    return data 




def get_sepsis_score(test, model):

#     print("data dhas been built .......")
    # Make some predictions and put them alongside the real TTE and event indicator values
#     print(test)

    scores= model.predict(test)
    
    if len(scores)<10:
        scores =round(sum(scores)/len(scores), 3) 
    else: 
        scores =round(sum([val for val in scores[:-10]])/10, 3) 
    
    
    labels = int(scores>0.7) #.astype(int)

    return scores, labels



