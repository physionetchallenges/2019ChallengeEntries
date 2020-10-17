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
                processed_data[i][j*2]=-3000
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

        matrix=np.ones(1248)*(-5000)
        if (i>=7): 
            matrix[0:544]=whole_train[i-7:i+1,0:68].flatten()
        else:
            matrix[(544-(i+1)*68):544]=whole_train[0:i+1,0:68].flatten()

        x_mean=np.nanmean(whole_train[:,0:68],axis=0)
        x_std=np.nanstd(whole_train[:,0:68],axis=0)+0.01
        x_norm = np.nan_to_num((whole_train[i,0:68] - x_mean) / x_std)
        whole_train_normed=(whole_train[:,0:68]-x_mean)/x_std
        if (i>=7): 
            matrix[544:1088]=whole_train_normed[i-7:i+1,0:68].flatten()
        else:
            matrix[(1088-(i+1)*68):1088]=whole_train_normed[0:i+1,0:68].flatten()

        matrix[1088:1156]=x_std
        matrix[1156:1168]=whole_train[i,68:].flatten()
        matrix[1168:1248]=np.sum(whole_train[:,:][whole_train[:,:]==-3000],axis=0)/(-3000.0)/float(whole_train.shape[0])
        i=i+1
    return matrix
