#!/usr/bin/env python
import sys
import os
sys.path.append('model')
import numpy as np
import tarfile
from config import *
from build import *
from manager import *

def get_sepsis_score(data, model):
    score = model.inference(data)
    label = score >= 0.6
    return score, label

def load_sepsis_model():
    #weight_unzip()
    cfg = Config()
    model = Build(cfg)
    manager = Manager(cfg, model)
    return manager


def weight_unzip():
    directory = './model/weight/'
    for s in os.listdir(directory):
        if 'gz' in s:
            filename = s
    x = tarfile.open(directory+filename)
    x.extractall(directory)
    x.close()
