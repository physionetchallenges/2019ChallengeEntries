#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob
from torch.utils.data import Dataset
import copy
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self,input_size,hidden_size,hidden_depth):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.hidden_depth, bidirectional=False, dropout=0)
        self.fc1 = nn.Linear(1*self.hidden_size, 2)

    def forward(self, x):
        h0 = torch.zeros(self.hidden_depth * 1, 1, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.hidden_depth * 1, 1, self.hidden_size).to(x.device)

        x = x.view(-1, 1, self.input_size)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc1(x)
        x = x.view(-1, 2)
        x = x[-1, :].view(1, 2)
        return x, F.softmax(x, dim=1)



def get_sepsis_score(data, model):
    data = pd.DataFrame(data)
    data = data.fillna(method='ffill')
    data = data.fillna(0).values
    norm = [2.800e+02, 1.000e+02, 5.000e+01, 3.000e+02, 3.000e+02, 3.000e+02, 1.000e+02,
            1.000e+02, 1.000e+02, 5.500e+01, 4.000e+03, 7.930e+00, 1.000e+02, 1.000e+02,
            9.961e+03, 2.680e+02, 3.833e+03, 2.790e+01, 1.450e+02, 4.660e+01, 3.750e+01,
            9.880e+02, 3.100e+01, 9.800e+00, 1.880e+01, 2.750e+01, 4.960e+01, 4.400e+02,
            7.170e+01, 3.200e+01, 2.500e+02, 4.400e+02, 1.760e+03, 2.322e+03, 1.000e+02,
            1.000e+00, 1.000e+00, 1.000e+00, 2.399e+01, 3.360e+02]
    data = data/norm
    data = torch.Tensor(data).float()
    threshold = 0.10
    _, probs = model(data)
    probs = probs[:, 1]
    argmax = (probs > threshold)
    return probs.data.numpy()[0], argmax.data.numpy()[0]

def load_sepsis_model():
    modelpth = './model_1561740354_cv_0_16_0.09277561709115473'
    model = Model(40,100,2)
    model.load_state_dict(torch.load(modelpth))
    model.eval()
    return model
