#!/usr/bin/env python

import ltsm
import numpy as np


def make_n(data, n):
    data_len = len(data)
    if data_len == n:
        return data
    if data_len > n:
        return data[-n:]
    r = data[-1]
    r_reshaped = r.reshape(1, data.shape[1])
    for i in range(data_len, n):
        data = np.append(data, r_reshaped, axis=0)
    return data


def get_sepsis_score(data, model):
    data = make_n(data, 6)
    # print("data = {}".format(data))
    features = ltsm.get_features(data)
    # print("features = {}".format(features))
    score = model.predict(features)[0][0]
    label = score > 0.03
    print("score = {}, label = {}".format(score, label))

    return score, label


def load_sepsis_model():
    m = ltsm.get_model()
    return m

