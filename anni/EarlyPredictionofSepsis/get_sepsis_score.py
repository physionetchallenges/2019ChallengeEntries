import lightgbm as lgbm
import os
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib


def load_sepsis_model():
    return joblib.load('./models/lgbm_0.bin')


def get_sepsis_score(values, model):
    thr = 0.29
    # models_root = '/home/yfdai/sepsis/physionet-v1/data/logs/2019-05-23-09-57-51'
    # models_paths = [os.path.join(models_root, m) for m in os.listdir(models_root) if m.endswith('.bin')]
    # models_paths.sort()

    # scores = np.zeros((len(values), ))
    # for path in models_paths:
    #     model = load_model(path)
    #     scores += model.predict_proba(values)[:, 1]

    # scores = scores / len(models_paths)
    # labels = np.where(scores > thr, 1, 0)

    scores = model.predict_proba(values)[:, 1]
    labels = np.where(scores > thr, 1, 0)

    return (scores[-1], labels[-1])




