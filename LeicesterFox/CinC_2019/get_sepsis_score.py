#!/usr/bin/env python3

import sys
import numpy as np
from tensorflow.keras.models import load_model
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_sepsis_score(data,model):
    model1=model[0]
    model2=model[1]
    max_v=[280,100,50,300,300,300,100,100,100,55,4000,7.93,100,100,9961,268,3833,27.9,145,46.6,37.5,988,31,9.8,18.8,27.5,49.6,440,71.7,32,250,440,1760,2322,100,1,1,1,23.99,336]
    min_v=[2.00E+01,2.00E+01,2.09E+01,2.00E+01,2.00E+01,2.00E+01,1.00E+00,1.00E+01,-3.20E+01,0.00E+00,-5.00E+01,6.62E+00,1.00E+01,2.30E+01,3.00E+00,1.00E+00,7.00E+00,1.00E+00,2.60E+01,1.00E-01,1.00E-02,1.00E+01,2.00E-01,2.00E-01,2.00E-01,1.00E+00,1.00E-01,1.00E-02,5.50E+00,2.20E+00,1.25E+01,1.00E-01,3.40E+01,1.00E+00,1.40E+01,0.00E+00,0.00E+00,0.00E+00,-5366.86,1.00E+00]
    max_v=np.asarray(max_v)
    min_v=np.asarray(min_v)
 
    l=len(data)
    M0=data;
    
    M0=(M0-min_v)/(max_v-min_v)
    for j in range(0,len(M0)):
        if j<12:
            temp=np.zeros((12,40));
            temp[np.maximum(j-12,0):j,:]=M0[np.maximum(j-12,0):j,0:40]; 
        else:
            temp=np.zeros((12,40));
            temp=M0[np.maximum(j-12,0):j,0:40];
    X = temp;

    X=np.nan_to_num(X)
#    X = X.reshape(len(X),11,11,1);
    X = X.reshape(1,12,40);
#    model = load_model('my_model_v8.h5')
#    labels= model.predict_classes(X)
    scores1= model1.predict_proba(X)
    scores1=scores1[:,1];
    ########################################################## MODEL 2
    
    Empty_M=np.zeros((66,1));
    feature_select=np.vstack((0,1,2,3,4,5,6,7,21,35,39));
    l=len(data)
    M0=data;
    X=np.zeros((l,121));    
#    for j in range(0,len(M0)):
    j=len(M0)-1;
    if j<5:
        temp=Empty_M
        for k in range(0,j+1):
            temp[11*k:11*k+11]=M0[k,feature_select];
    else:
         temp=Empty_M
         for k in range(j-5,j+1):
           temp[11*(k-j+5):11*(k-j+5)+11]=M0[k,feature_select];       
    temp =  np.vstack((temp, (temp[0:11]-temp[11:22])))
    temp =  np.vstack((temp, (temp[11:22]-temp[22:33])))
    temp =  np.vstack((temp, (temp[22:33]-temp[33:44])))
    temp =  np.vstack((temp, (temp[33:44]-temp[44:55])))
    temp =  np.vstack((temp, (temp[44:55]-temp[55:66])))
#    X[j,:]=np.transpose(temp);
    X=np.transpose(temp);
    
    X=np.nan_to_num(X)
#    X = X.reshape(len(X),11,11,1);
    X = X.reshape(1,11,11,1);
#    model = load_model('my_model_v8.h5')
#    labels= model.predict_classes(X)
    scores2= model2.predict_proba(X)
    scores2=scores2[:,1];
    
    scores=(scores1+scores2)/2;
    labels=scores>0.5;
    
    return (scores, labels)

def load_sepsis_model():
    model1 = load_model('my_model_RNN_v4.h5')
    model2 = load_model('my_model_v6.h5')
    model=[model1,model2]
    return (model)