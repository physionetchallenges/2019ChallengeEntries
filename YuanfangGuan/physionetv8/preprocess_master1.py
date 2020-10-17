#!/usr/bin/perl
#

import numpy as np
import math

def feature(whole_train): ## -3000
    i=whole_train.shape[0]-1
    while(i<whole_train.shape[0]):

        matrix_1=np.ones(1248)*(-5000)
        if (i>=7): 
            matrix_1[0:544]=whole_train[i-7:i+1,0:68].flatten()
        else:
            matrix_1[(544-(i+1)*68):544]=whole_train[0:i+1,0:68].flatten()

        x_mean=np.nanmean(whole_train[:,0:68],axis=0)
        x_std=np.nanstd(whole_train[:,0:68],axis=0)+0.01
        x_norm = np.nan_to_num((whole_train[i,0:68] - x_mean) / x_std)
        whole_train_normed=(whole_train[:,0:68]-x_mean)/x_std
        if (i>=7): 
            matrix_1[544:1088]=whole_train_normed[i-7:i+1,0:68].flatten()
        else:
            matrix_1[(1088-(i+1)*68):1088]=whole_train_normed[0:i+1,0:68].flatten()

        matrix_1[1088:1156]=x_std
        matrix_1[1156:1168]=whole_train[i,68:].flatten()
        matrix_1[1168:1248]=np.sum(whole_train[:,:][whole_train[:,:]==-3000],axis=0)/(-3000.0)/float(whole_train.shape[0])

        matrix_2=np.ones(2256)*(-3000)
        matrix_2[0:1168][matrix_1[0:1168]!=-5000]=matrix_1[0:1168][matrix_1[0:1168]!=-5000]
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
        matrix_2[1168:1712]=the_long.flatten()
        matrix_2[1712:2256]=the_time.flatten()


    return matrix_1,matrix_2


        i=i+1
    return matrix_1
