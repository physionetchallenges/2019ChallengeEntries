import numpy as np
from sklearn.externals import joblib
import lightgbm as lgb



def get_sepsis_score(data, model):

    data1=data[-1]
    data1=data1.reshape(1,-1)
    score1 = model.predict(data1,num_iteration=model.best_iteration)
    label=score1>0.6016
    return score1, label



def load_sepsis_model():
    clf=joblib.load("gbm_v300_df=20_nl=20LR=2.pkl")
    return clf
