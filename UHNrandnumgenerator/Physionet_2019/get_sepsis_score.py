#!/usr/bin/env python

import numpy as np
import os
import torch 
from model import TCN

location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) + '/' + 'vals.npy'
vals = np.load(location)
maxs = np.expand_dims(vals[:,1], axis=0)
mins = np.expand_dims(vals[:,2], axis=0)


def get_sepsis_score(data, model):
    markers = np.isnan(data).astype(int)
    data = (data - mins) / (maxs - mins)

    for i in range(len(data)):
        if i == 0:
            data[i] = np.nan_to_num(data[i]) - np.isnan(data[i]).astype(int)
        else:
            data[i] = np.nan_to_num(data[i]) + (np.isnan(data[i]).astype(int) * data[i-1])
        
    data =  np.concatenate((markers, data), axis=1)
    
    tensor = torch.from_numpy(data).unsqueeze(0).float()

    score = model(tensor.permute(0,2,1))

    if score.dim() == 0:
        score = score
    else:
        score = score[-1]


    score = np.float64(score.data.numpy())
    label = score > 0.6
    return score, label

def load_sepsis_model():
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))) + '/' + 'state_dict'

    model = TCN(80, 1, [64, 64], fcl=32,  attention=2, kernel_size=2, dropout=0)

    model.load_state_dict(torch.load(location, map_location='cpu'))
    model.eval()

    return model

