#authors: Anamika Paul Rupa, Al Amin, Sanjay Purushotham
#email:  rupa3@umbc.edu

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from joblib import dump, load

def get_sepsis_score(data,model):
    shape = data.shape[0] - 1
    scaler = load('scaler3labelnew')
    data = pd.DataFrame(data)
    d = data.fillna(method='ffill').values
    X = scaler.transform(d)
    X[np.isnan(X)] = 0
    scores0=model.predict_proba(X)[:, 0]
    scores1=model.predict_proba(X)[:, 1]
    scores2=model.predict_proba(X)[:, 2]
    m=max(scores0[shape],scores1[shape],scores2[shape])
    m2=min(scores0[shape],scores1[shape],scores2[shape])
    labels = 0
    if(m==scores2[shape]):
         return (m,1)
    else:
        return (m2, 0)

def load_sepsis_model():

    model=load('xgboost3labelnew.joblib')
    return model


