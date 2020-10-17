#!/usr/bin/env python
import numpy as np

import network
import preprocess as prepro

np.warnings.filterwarnings('ignore')


def get_sepsis_score(data, model):
    # parameters
    used_features = list(range(36)) + [38, 39]
    threshold = 0.07

    # prepare input data
    x = data[:, used_features]

    x_diff = prepro.get_diff(data, list(range(34)))
    x_sofa, _ = prepro.get_SOFA(data)
    x_qsofa, _ = prepro.get_qSOFA(data)
    x_reli = prepro.last_reliable(x)
    x = np.concatenate([x, x_diff, x_sofa, x_qsofa, x_reli], 1)

    # remove nans
    x[np.isnan(x)] = 0
    x = x[np.newaxis, :, :]

    # classify/predict
    y_pred_probs = np.zeros((5, np.shape(x)[1]))
    for n in range(5):
        y_pred_prob = model[n].predict(x, 1)
        y_pred_probs[n, :] = y_pred_prob[0, :, 1]

    y_pred_prob = np.mean(y_pred_probs, axis=0)
    # generate label
    y_pred = y_pred_prob > threshold

    current_score = y_pred_prob[-1]
    current_label = y_pred[-1]

    # only return last
    return current_score, current_label


def load_sepsis_model():
    models = []
    for n in range(5):
        model = network.get_lstm(129)
        model.load_weights(f"model{n}")
        models.append(model)
    return models
