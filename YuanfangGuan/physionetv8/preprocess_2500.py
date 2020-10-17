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

        matrix_3=np.ones(2256)*(-5000)
        if (i>=7): 
            matrix_3[0:544]=whole_train[i-7:i+1,0:68].flatten()
        else:
            matrix_3[(544-(i+1)*68):544]=whole_train[0:i+1,0:68].flatten()

        x_mean=np.nanmean(whole_train[:,0:68],axis=0)
        x_std=np.nanstd(whole_train[:,0:68],axis=0)+0.01
        x_norm = np.nan_to_num((whole_train[i,0:68] - x_mean) / x_std)
        whole_train_normed=(whole_train[:,0:68]-x_mean)/x_std
        if (i>=7): 
            matrix_3[544:1088]=whole_train_normed[i-7:i+1,0:68].flatten()
        else:
            matrix_3[(1088-(i+1)*68):1088]=whole_train_normed[0:i+1,0:68].flatten()

        matrix_3[1088:1156]=x_std
        matrix_3[1156:1168]=whole_train[i,68:].flatten()
        the_long=np.zeros((8,68))
        the_time=np.ones((8,68))*(-2500)
        jjj=0
        while (jjj<68):
            iii=0
            string=[]
            timestring=[]
            while (iii<whole_train.shape[0]):
                if ((whole_train[iii,jjj] == np.nan) or (whole_train[iii,jjj] == -2500)):
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
        matrix_3[1168:1712]=the_long.flatten()
        matrix_3[1712:2256]=the_time.flatten()


        matrix_6=np.ones(1440)*(-5000)
        if (i>=7): 
            matrix_6[0:640]=whole_train[i-7:i+1,:].flatten()
        else:
            matrix_6[(640-(i+1)*80):640]=whole_train[0:i+1,:].flatten()

        x_mean=np.nanmean(whole_train[:,:],axis=0)
        x_std=np.nanstd(whole_train[:,:],axis=0)+0.01
        x_norm = np.nan_to_num((whole_train[i,:] - x_mean) / x_std)
        whole_train_normed=(whole_train[:,:]-x_mean)/x_std
        if (i>=7): 
            matrix_6[640:1280]=whole_train_normed[i-7:i+1,:].flatten()
        else:
            matrix_6[(1280-(i+1)*80):1280]=whole_train_normed[0:i+1,:].flatten()

        i=i+1
        matrix_6[1280:1360]=x_std
        matrix_6[1360:1440]=np.sum(whole_train[i-7:i+1,:][whole_train[i-7:i+1,:]==-2500],axis=0)/(-2500.0)

    return matrix_3,matrix_6
