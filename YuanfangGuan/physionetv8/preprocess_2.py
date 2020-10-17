#!/usr/bin/perl
#

import numpy as np
import math
def avg():
    the_average={}
    AVG=open('median.txt','r')
    i=0
    for line in AVG:
        line=line.rstrip()
        table=line.split('\t')
        the_average[i]=float(table[1])
        i=i+1
    return the_average



def preprocess(data):
    average=avg()
#    print(data.shape)
    imax=data.shape[0]
    jmax=data.shape[1]
    processed_data=np.zeros((imax,jmax*2))

    i=0
    while (i<imax):
        j=0
        while (j<jmax):
            if (math.isnan(data[i][j])):
                #processed_data[i][j*2]=-3000
                processed_data[i][j*2]=average[j]
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

        matrix=np.ones(2256)*(-3000)
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
        i=i+1
    the_long=np.zeros((8,68))
    the_time=np.ones((8,68))*(-3000)
    jjj=0
    while (jjj<68):
        iii=0
        string=[]
        timestring=[]
        while (iii<whole_train.shape[0]):
            if ((whole_train[iii,jjj] == np.nan) or (whole_train[iii,jjj] == -3000)):
#                    string.append(0)
                pass
            else:
                diff_tmp=whole_train[iii,jjj]
                string.append(diff_tmp)
                diff_tmp=iii-whole_train.shape[0]
                timestring.append(diff_tmp)
            iii=iii+1
        string=np.asarray(string)
        timestring=np.asarray(timestring)
        if (string.shape[0]>7):
            the_long[:,jjj]=string[(string.shape[0]-8):]
            the_time[:,jjj]=timestring[(timestring.shape[0]-8):]
        else:
            the_long[(8-string.shape[0]):,jjj]=string
            the_time[(8-string.shape[0]):,jjj]=timestring
        
        jjj=jjj+1
    matrix[1168:1712]=the_long.flatten()
    matrix[1712:2256]=the_time.flatten()


    return matrix
