#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:33:28 2019

@author: brou
"""

from network import *
import numpy as np, os, os.path
import torch
from torch.autograd import Variable
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICE"] = "0"

def get_sepsis_score(data,models): 
    model1, model2, model3, model4 = models
    model1, model2, model3, model4 = model1.cuda(), model2.cuda(), model3.cuda(), model4.cuda()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    # Les signaux vitaux que l'on va utiliser
    vital_headers = [0,1,2,3,4,5,6,7]
    demo_headers = [34,35,36,37,39,38]
    physio_headers = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
    
    #Minimum et maximum pour chaque signal vital, pour normalisation
    maxi_vital = np.array([300. , 100. ,  75., 400. , 400. , 400. ,  150. ,  150. ])
    mini_vital = np.array([0. , 0. , 0., 0., 0. , 0. ,  0. , 0. ])
    
    maxi_demo = np.array([100.,1.,1.,1.,500,0])
    mini_demo = np.array([0.,0.,0.,0.,0.,-240])
    
    maxi_physio = np.array([150, 75, 5000, 9, 125, 125, 10000, 300, 4000, 30, 160, 55, 42.5, 1000, 35, 11, 22.5, 30, 55, 475, 80, 38, 275, 450,1850, 2500])
    mini_physio = np.array([-40,  0, -75,  5,  5,  17.5, -1, -1,  5,  -1,  24,  -1, -1,  5,  -1,  0.005,  0.005,  0.5, -1,  -1,  4,  1,  10.,  -1, 30, -1])
    
    nb_vital_features = len(vital_headers)  
    nb_demo_features = len(demo_headers)
    nb_physio_features = len(physio_headers)
    nb_features = nb_vital_features + nb_demo_features + nb_physio_features
    
    seq_len = data.shape[0]
    vital_data = np.zeros((seq_len,nb_features + nb_physio_features + 5*nb_vital_features))
    for header in range(nb_vital_features):
        column = data[:,vital_headers[header]]
        #On normalise
        column = (column - mini_vital[header])/(maxi_vital[header] - mini_vital[header])
        
        column[np.isnan(column)] = -1
        vital_data[:seq_len,header] = column
        for k in range(1,6):
            vital_data[k:seq_len,k*nb_vital_features + header] = column[:-k]
    for header in range(nb_demo_features):
        column = data[:,demo_headers[header]]        

        column = (column - mini_demo[header])/(maxi_demo[header] - mini_demo[header])
        
        
        column[np.isnan(column)] = -1
        vital_data[:seq_len,header+6*nb_vital_features] = column
    for header in range(nb_physio_features):
        column = data[:,physio_headers[header]]        

        column = (column - mini_physio[header])/(maxi_physio[header] - mini_physio[header])
        vital_data[:seq_len,header+6*nb_vital_features+nb_demo_features+nb_physio_features] = (~np.isnan(column)).astype(float) 
        column[np.isnan(column)] = -1
        vital_data[:seq_len,header+6*nb_vital_features+nb_demo_features] = column             

    vital_data = Variable(torch.tensor(vital_data).float().unsqueeze(0))
    vital_data = vital_data.cuda()
    seq_len = torch.as_tensor([seq_len], dtype=torch.int64,device='cpu')
    
    with torch.no_grad() :
        pred1, y_pred1 = model1(vital_data, seq_len)
        y_pred1 = y_pred1[:,:,5,:]
        scores1 = F.softmax(y_pred1,dim=2)

        pred2, y_pred2 = model2(vital_data, seq_len)
        y_pred2 = y_pred2[:,:,5,:]
        scores2 = F.softmax(y_pred2,dim=2)        

        pred3, y_pred3 = model3(vital_data, seq_len)
        y_pred3 = y_pred3[:,:,5,:]
        scores3 = F.softmax(y_pred3,dim=2)      

        pred4, y_pred4 = model4(vital_data, seq_len)
        y_pred4 = y_pred4[:,:,5,:]
        scores4 = F.softmax(y_pred4,dim=2)      
        
        scores = (scores1 + scores2 + scores3 + scores4)/4.
        (scores, pred_labels) = scores.max(dim=2)
        scores = (1 - scores)*(1- pred_labels.float()) + pred_labels.float()*scores
        score = scores[0][-1]
        label = int(score > 0.6)
    
    return(score, label)
    

def load_sepsis_model():
    model1 = LSTM(8, 6,26, 32, 64,6,2)
    model1.load_state_dict(torch.load('parameters1.pt', map_location='cpu'))
    
    model2 = LSTM(8, 6,26, 32, 64,6,2)
    model2.load_state_dict(torch.load('parameters2.pt', map_location='cpu'))
    
    model3 = LSTM(8, 6,26, 32, 64,6,2)
    model3.load_state_dict(torch.load('parameters3.pt', map_location='cpu'))

    model4 = LSTM(8, 6,26, 32, 64,6,2)
    model4.load_state_dict(torch.load('parameters4.pt', map_location='cpu'))
    return (model1, model2, model3, model4)