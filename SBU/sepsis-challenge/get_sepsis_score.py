#!/usr/bin/env python3

import numpy as np
import pickle as pkl
from keras.preprocessing.sequence import pad_sequences
import os

def impute_last_seen(data):
    x_mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777, 60.8711, 0.5435,
        0.0615, 0.0727, -59.6769, 28.4551])
    last_known_time = np.zeros(x_mean.shape)
    last_known_val = x_mean

    values = data.copy()
    values_matrix = np.zeros((values.shape[0], 2 * values.shape[1]))

    for i in range(values.shape[0]):
        is_nan = np.where(np.isnan(values[i]),
                    np.ones(last_known_val.shape),
                    np.zeros(last_known_val.shape))

        values[i] = np.where(np.isnan(values[i]),
                            last_known_val,
                            values[i])

        last_known_time = last_known_time * is_nan
        last_known_time += is_nan
        last_known_val = values[i]
        values_matrix[i] = np.hstack((values[i], last_known_time))
    return values_matrix

def get_sepsis_score(data, model):

    #print(data.shape)
    x = data[:, 0:40]
    #print(x.shape)
    x = impute_last_seen(x).reshape(1, x.shape[0], 80)
    #print(x.shape)
    x = pad_sequences(x, 20, padding='pre')
    #print(x.shape)
    x = x.reshape(-1, 20 * 80)
    #print(x.shape)
    #print("LINE")
    scores = model.predict_proba(x)[:, 1]
    labels = (scores > 0.0420023)
    #scores = np.exp(scores) / (1.0 + np.exp(scores))
    #print(scores, labels)
    #scores, labels = np.asscalar(scores), np.asscalar(labels)
    return (scores[-1], labels[-1])

def load_sepsis_model():
    model = pkl.load(open(os.path.join(os.getcwd(), 'model_seq.pkl'), 'rb'))
    return model
