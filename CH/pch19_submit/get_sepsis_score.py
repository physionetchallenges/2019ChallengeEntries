#!/usr/bin/env python

import numpy as np
from tensorflow import keras

MODEL_FILE_NAME = 'weights00000005-0.4603-0.8046-0.4603.h5'

PROB_MOD = 0.35

def get_sepsis_score(data, model):

    test_example = np.array([data])

    time_series_count = np.array([data.shape[0]])
    test_example = repair_data(test_example, time_series_count)

    prediction_probabs = model.predict_on_batch(test_example)
    prediction_class = np.round(prediction_probabs + PROB_MOD)

    score = prediction_probabs[0,-1,0]
    label = prediction_class[0,-1,0]

    return score, label

def load_sepsis_model():
    model = keras.models.load_model(MODEL_FILE_NAME)
    return model

def repair_data(data, time_series_count):
    masks = []
    for record_idx, record in enumerate(data):
        mask = np.zeros((record.shape[0], record.shape[1]), dtype=np.float32)
        for col_idx in range(record.shape[1]):
            # find two numbers on edges
            upper_edge_idx = 0 if not np.isnan(record[0, col_idx]) else -1
            lower_edge_idx = -1
            for row_idx in range(1, time_series_count[record_idx]):
                if row_idx <= upper_edge_idx:
                    continue
                if not np.isnan(record[row_idx, col_idx]):
                    lower_edge_idx = row_idx

                    # replace nan with interpolation between val[upper_edge_idx] and val[lower_edge_idx],
                    # if upper_edge_idx is -1, copy values from val[lower_edge_idx] till top,
                    # if lower_edge_idx is -1, copy values from val[upper_edge_idx] till bottom

                    if upper_edge_idx == -1:
                        for i in range(0, lower_edge_idx):  # replace all nans from 0 to lower_edge_idx-1 with val[lower_edge_idx]
                            record[i, col_idx] = record[lower_edge_idx, col_idx]
                            mask[i, col_idx] = 1
                    elif upper_edge_idx >= 0:
                        for i in range(upper_edge_idx + 1, lower_edge_idx):  # replace all nans from upper_edge_idx+1 to lower_edge_idx-1 with interpolation
                            record[i, col_idx] = (record[upper_edge_idx, col_idx] + record[lower_edge_idx, col_idx]) / 2
                            mask[i, col_idx] = 1
                    upper_edge_idx = lower_edge_idx

            if upper_edge_idx == -1 and lower_edge_idx == -1:  # all are nans, then fill with 0
                for i in range(time_series_count[record_idx]):
                    record[i, col_idx] = 0
                    mask[i, col_idx] = 1

            if np.isnan(record[time_series_count[record_idx] - 1, col_idx]):  # replace all nans from upper_edge_idx to [-1] with val[upper_edge_idx]
                for i in range(upper_edge_idx, time_series_count[record_idx]):
                    record[i, col_idx] = record[upper_edge_idx, col_idx]
                    mask[i, col_idx] = 1

        masks.append(mask)
    masks = np.array(masks)
    data = np.concatenate((data, masks), axis=2)

    return data
