#!/usr/bin/env python

import numpy as np
import pandas as pd
import math
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection 
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        outs = self.fc(out)
        outs = F.log_softmax(outs, dim=1)
        return outs

def reset_data(data, del_ratio=0.4):
    column_list = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
        'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
        'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
        'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1',
        'Unit2', 'HospAdmTime', 'ICULOS']
    data = pd.DataFrame(data,columns=column_list)
    
    zero_ratio_table = {'DBP': 7411, 'EtCO2': 37120, 'Bilirubin_direct': 38279, 'Lactate': 27843,
                        'TroponinI': 33283, 'PTT': 20098, 'Fibrinogen': 35821, 'BaseExcess': 27915, 'FiO2': 22529,
                        'pH': 21401, 'PaCO2': 21980, 'SaO2': 27248, 'AST': 25979, 'Alkalinephos': 26163,
                        'Bilirubin_total': 26088, 'Calcium': 5339, 'Magnesium': 4931, 'Phosphate': 12015,
                        'O2Sat': 18, 'Hgb': 2448, 'Temp': 284, 'HCO3': 20119, 'BUN': 2018, 'Chloride': 18925,
                        'Creatinine': 2049, 'Glucose': 1580, 'Potassium': 1867, 'Hct': 2317, 'WBC': 2625,
                        'Platelets': 2577, 'SBP': 282, 'Resp': 71, 'MAP': 104, 'HR': 5}
    del_list = []
    for i in list(zero_ratio_table.keys()):
        cnum = zero_ratio_table[i] / 40336
        if cnum > del_ratio:
            del_list.append(i)
    for del_c in del_list:
        data = data.drop([del_c], axis=1)
    
    data = data.fillna(method='pad').fillna(method='bfill').fillna(0)

    global_index_list = ['Age','HospAdmTime']
    global_max_list = {'Age':100 ,'HospAdmTime':23.99}
    global_min_list = {'Age':14 ,'HospAdmTime':-5366.86}
    for index in global_index_list:
        data[index] = (data[index] - global_min_list[index])/(global_max_list[index] - global_min_list[index])
        
    index_list = list(data.columns)
    for index in index_list[0:-2]:
        d_max = data[index].max()
        d_min = data[index].min()
        d_std = d_max - d_min
        if d_std != 0:
            data[index] = (data[index] - d_min)/d_std
            
    data = data.values.astype(np.float32)[np.newaxis,:,:]
    data = torch.from_numpy(data)
    return data

def get_sepsis_score(data, model):
    score = 0.0
    label = 0
    if len(data) > 2:
        data = reset_data(data)
        outputs = model(data)
        probability, predicted = torch.max(outputs.data, 2)
        probability = torch.pow(math.e, probability)
        scores = probability[0].numpy()
        labels = predicted[0].numpy().astype(np.int)
        if labels.mean() < 0.006:
            labels = np.zeros(labels.shape)
        score = float(scores[-1])
        label = labels[-1]
        
    return score, label

def load_sepsis_model():
    model = BiRNN(24, 32, 2, 2)
    model.load_state_dict(torch.load('./rnn.pkl'))
    model.eval()
    return model
