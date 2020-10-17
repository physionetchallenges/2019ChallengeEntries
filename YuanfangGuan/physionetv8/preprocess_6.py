#!/usr/bin/perl
#

import numpy as np
import math
def preprocess(data):
#    print(data.shape)
    imax=data.shape[0]
    jmax=data.shape[1]
    processed_data=np.zeros((imax,jmax*2))
    i=0
    while (i<imax):
        j=0
        while (j<jmax):
            if (math.isnan(data[i][j])):
                processed_data[i][j*2]=-2500
            else:
                processed_data[i][j*2]=data[i][j]
            if (math.isnan(data[i][j])):
                processed_data[i][j*2+1]=0
            else:
                processed_data[i][j*2+1]=1
            j=j+1
        i=i+1
    return processed_data
	

def feature(whole_train):
    i=whole_train.shape[0]-1
    while(i<whole_train.shape[0]):
        matrix=np.ones(1440)*(-5000)
        if (i>=7): 
            matrix[0:640]=whole_train[i-7:i+1,:].flatten()
        else:
            matrix[(640-(i+1)*80):640]=whole_train[0:i+1,:].flatten()

        x_mean=np.nanmean(whole_train[:,:],axis=0)
        x_std=np.nanstd(whole_train[:,:],axis=0)+0.01
        x_norm = np.nan_to_num((whole_train[i,:] - x_mean) / x_std)
        whole_train_normed=(whole_train[:,:]-x_mean)/x_std
        if (i>=7): 
            matrix[640:1280]=whole_train_normed[i-7:i+1,:].flatten()
        else:
            matrix[(1280-(i+1)*80):1280]=whole_train_normed[0:i+1,:].flatten()

        i=i+1
    matrix[1280:1360]=x_std
    matrix[1360:1440]=np.sum(whole_train[i-7:i+1,:][whole_train[i-7:i+1,:]==-2500],axis=0)/(-2500.0)
    return matrix
