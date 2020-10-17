#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.externals import joblib

def get_sepsis_score(data, model):
    indices = list(range(0,40))
    df = np.take(data[-1],indices)
    df = np.nan_to_num(df)
    df = df.reshape(1,-1)
    prob = model.predict_proba(df)[:,1]
    score = model.predict(df)
    return prob, score

def load_sepsis_model():
    model = joblib.load('finalized_model11.sav')
    return model
