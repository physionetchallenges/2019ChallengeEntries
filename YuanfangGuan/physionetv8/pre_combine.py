#!/usr/bin/perl
#

import numpy as np
import math
import copy
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
#    print(data.shape)
    imax=data.shape[0]
    jmax=data.shape[1]
    processed_data=np.zeros((imax,jmax*2))
    processed_data_median=np.zeros((imax,jmax*2))
    the_average=avg()

    i=0
    while (i<imax):
        j=0
        while (j<jmax):
            if (math.isnan(data[i][j])):
                processed_data[i][j*2]=-3000
                processed_data_median[i][j*2]=the_average[j]
            else:
                processed_data[i][j*2]=data[i][j]
                processed_data_median[i][j*2]=data[i][j]

            if (math.isnan(data[i][j])):
                processed_data[i][j*2+1]=0
                processed_data_median[i][j*2+1]=0
            else:
                processed_data[i][j*2+1]=1
                processed_data_median[i][j*2+1]=1
            j=j+1
        i=i+1
    processed_data_2500=copy.copy(processed_data)
    processed_data_2500[processed_data==-3000]=-2500
    return processed_data,processed_data_median,processed_data_2500
	

