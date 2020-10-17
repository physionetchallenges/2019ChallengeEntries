#!/usr/bin/env python

import numpy as np
from keras.models import load_model

def get_sepsis_score(data, model):
    max_val = np.array([280.0, 100.0, 42.22, 299.0, 300.0, 298.0, 100.0, 100.0, 100.0, 55.0, 10.0, 7.93, 100.0, 100.0, 9961.0, 268.0, 3833.0, 
        27.9, 145.0, 46.6, 37.5, 988.0, 31.0, 9.7, 18.8, 27.5, 49.6, 440.0, 71.7, 32.0, 250.0, 440.0, 1760.0, 2322.0, 100.0, 1.0, 1.0, 1.0, 23.99])
    min_val = np.array([20.0, 20.0, 20.9, 20.0, 20.0, 20.0, 1.0, 10.0, -32.0, 0.0, -50.0, 6.62, 10.0, 24.0, 3.0, 1.0, 7.0, 1.0, 26.0, 0.1, 0.01, 
        10.0, 0.2, 0.2, 0.2, 1.0, 0.1, 0.01, 5.5, 2.2, 12.5, 0.1, 34.0, 2.0, 15.0, 0.0, 0.0, 0.0, -5366.86])
    mean_val = np.array([84.83, 97.22, 37, 122.3, 80.56, 62.18, 18.76, 33.02, -0.6622, 24.08, 0.525, 7.379, 41.09, 91.78, 310.58, 24.12, 108.6, 7.91, 105.78, 1.45,
        2.39, 135.68, 2.567, 2.04, 3.56, 4.14, 2.368, 8.516, 30.71, 10.50, 41.01, 11.71, 289.58, 198.47, 62.43, 0.566, 0.5, 0.5, -55.27])

    data = data[:,0:-1] #drop last feature
    data = np.reshape(data,(1,-1,39))
    #print(data[0,0,0])
    #imputation and normalize
    #print(data)
    for col in range(0,data.shape[2]):
        if np.isnan(data[0,-1,col]):
            data[0,-1,col] = min_val[col]
        data[0,-1,col] = (data[0,-1,col] - min_val[col])/(max_val[col] - min_val[col])

    #print(data)

    score = model.predict(data)[0,0]

    if score >=0.90:
        label = 1
    else:
        label = 0
    #print(score, label)

    return score, label

def load_sepsis_model():
    return load_model('model_3.h5')
