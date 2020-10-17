#!/usr/bin/env python

import numpy as np

from experiments.baseline_sklearn.baseline_sklearn import get_sepsis_score as get_sepsis_score_sklearn

def get_sepsis_score(data, model_file):
    return get_sepsis_score_sklearn(data, model_file)

def load_sepsis_model():
    ### LR ensemble
    return 'experiments/models/lr_for_submit.pkl'
