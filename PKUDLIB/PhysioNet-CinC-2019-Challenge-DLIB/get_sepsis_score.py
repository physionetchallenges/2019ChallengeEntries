#!/usr/bin/env python

'''
April 2019 by WHX
'''

import numpy as np
from architecture_decom import Graph
from hyperparams import Hyperparams_decom as hp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

g = Graph('test')
sess = tf.Session()
    
def to_onehot(x):
    tx = np.zeros((x.shape[0],40))
    for i in range(x.shape[0]):
        tx[i] = np.concatenate((x[i][:35], x[i][38:], x[i][35:38]), axis = 0)
        if np.isnan(tx[i][37]):
            tx[i][37] = 6
        if np.isnan(tx[i][38]):
            tx[i][38] = 6
        else: tx[i][38] += 2
        if np.isnan(tx[i][39]):
            tx[i][39] = 6
        else: tx[i][39] += 4
    return tx

def normal(x):
    mean = np.array([ 84.62099863,  97.18563716,  36.97851691, 123.63536038,  82.31340811,
                      63.72771958,  18.7357924,   33.03678896,  -0.92714211,  24.07694944,
                      0.56834562,   7.37871633,  41.01402152,  92.62905819, 254.81618435,
                      23.95669531, 102.34700855,   7.56702007, 105.83017132,   1.51227632,
                      1.92099388, 137.09198367,   2.61639216,   2.05286981,   3.54646446,
                      4.13633335,   2.08844781,   8.50454466,  30.79102151,  10.43544968,
                      41.15130053,  11.42609303, 289.83134021, 195.65310378,  62.03080237,
                      -57.37519834,  27.09726212])

    
    std = np.array( [ 1.73429174e+01, 2.94834690e+00, 7.75290525e-01, 2.31953297e+01,
                      1.63262076e+01, 1.39031272e+01, 5.10213577e+00, 8.05546811e+00,
                      4.85622690e+00, 4.37741851e+00, 1.32578447e+01, 7.43878087e-02,
                      9.24682308e+00, 1.08789664e+01, 8.46110303e+02, 1.99948804e+01,
                      1.14410913e+02, 2.41515383e+00, 5.87124667e+00, 1.81299038e+00,
                      3.87522385e+00, 5.17920992e+01, 2.44204407e+00, 3.99731580e-01,
                      1.42404313e+00, 6.39129209e-01, 4.23213489e+00, 2.56186731e+01,
                      5.49186677e+00, 1.96971680e+00, 2.59516109e+01, 7.16970639e+00,
                      1.54006629e+02, 1.03249487e+02, 1.63643200e+01, 1.46643244e+02,
                      2.91155288e+01])
    px = x[:,:,:37]
    px = np.where(px!=0, (px-mean)/std, 0)
    x = np.concatenate((px, x[:,:,37:]), axis = -1)
    return x
def zero_padding(x):
    x = to_onehot(x)    
    if x.shape[0] < hp.timelen:
        xz = np.zeros((hp.timelen-x.shape[0], hp.event_nums))
        x = np.concatenate((xz,x), axis = 0)
        
    x = np.nan_to_num(np.reshape(x, (1, hp.timelen, hp.event_nums)))
    x = normal(x)
    return x

def get_sepsis_score(data, model):
    data = zero_padding(data[max(data.shape[0]-hp.timelen,0):])
    scores = sess.run(g.preds, {g.x:data, g.T:False})[0]
    return scores, int(scores>0.02)

def load_sepsis_model():
    saver = tf.train.Saver()
    saver.restore(sess, './model/base')
    return None
