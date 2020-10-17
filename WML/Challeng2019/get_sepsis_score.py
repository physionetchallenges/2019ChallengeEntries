#!/usr/bin/env python

import numpy as np
from sklearn.externals import joblib

x_mean = np.array([
    83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
    66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
    0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
    22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
    0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
    4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
    38.9974, 10.5585,  286.5404, 198.6777])
x_std = np.array([
    17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
    14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
    6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
    19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
    1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
    0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
    29.8928, 7.0606,  137.3886, 96.8997])
c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

def get_sepsis_score(data, model):
    
    values=data
    t=len(data)  #time
    
    for j in range(34):       
        where_are_nan = np.isnan(values[:,j])
        values[:,j][where_are_nan]=x_mean[j]    
        
    for k in range(34,40):
        where_are_nan = np.isnan(values[:,k])
        values[:,k][where_are_nan]=c_mean[k-34]   
        
    data1=values
    
    x = data1[:, 0:34]
    c = data1[:, 34:40]
    
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    c_norm = np.nan_to_num((c - c_mean) / c_std)
    
    prepro_data = np.concatenate((x_norm, c_norm),axis=1)
#              
    phy_signal=prepro_data[:,0:7]    
    patient_label1=prepro_data[:,34:36]
    patient_label2=prepro_data[:,38:40]
 
    feature=np.concatenate((np.array(phy_signal),np.array(patient_label1),np.array(patient_label2)),axis=1)
    
    #获取预测概率
    cls=model    
    scores=cls.predict_proba(feature)
    score=scores[:,1]
    label = cls.predict(feature)
    
    score=score[t-1]
    label=label[t-1]

    return score, label

def load_sepsis_model():
    cls = joblib.load('ensemble_cls.pkl')
    return cls


#
#def get_sepsis_score(data, model):
#    x_mean = np.array([
#        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
#        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
#        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
#        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
#        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
#        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
#        38.9974, 10.5585,  286.5404, 198.6777])
#    x_std = np.array([
#        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
#        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
#        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
#        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
#        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
#        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
#        29.8928, 7.0606,  137.3886, 96.8997])
#    c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
#    c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])
#
#    x = data[-1, 0:34]
#    c = data[-1, 34:40]
#    x_norm = np.nan_to_num((x - x_mean) / x_std)
#    c_norm = np.nan_to_num((c - c_mean) / c_std)
#
#    beta = np.array([
#        0.1806,  0.0249, 0.2120,  -0.0495, 0.0084,
#        -0.0980, 0.0774, -0.0350, -0.0948, 0.1169,
#        0.7476,  0.0323, 0.0305,  -0.0251, 0.0330,
#        0.1424,  0.0324, -0.1450, -0.0594, 0.0085,
#        -0.0501, 0.0265, 0.0794,  -0.0107, 0.0225,
#        0.0040,  0.0799, -0.0287, 0.0531,  -0.0728,
#        0.0243,  0.1017, 0.0662,  -0.0074, 0.0281,
#        0.0078,  0.0593, -0.2046, -0.0167, 0.1239])
#    rho = 7.8521
#    nu = 1.0389
#
#    xstar = np.concatenate((x_norm, c_norm))
#    exp_bx = np.exp(np.dot(xstar, beta))
#    l_exp_bx = pow(4 / rho, nu) * exp_bx
#
#    score = 1 - np.exp(-l_exp_bx)
#    label = score > 0.45
#
#    return score, label
#
#def load_sepsis_model():
#    return None