#!/usr/bin/env python

from sklearn.externals import joblib
from keras.models import load_model, Model
import numpy as np
from scipy import interpolate
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def summarize_pati(data, hours_sum):
    # To create the sliding windows
    rows_dummy = np.repeat(data, hours_sum - 1, axis=0)
    rows_dummy = rows_dummy[0:hours_sum - 1, :]

    data = np.append(rows_dummy, data, axis=0)

    num_rows_current, num_features = data.shape  # get new shape

    # sliding window
    sample = np.empty((num_rows_current - hours_sum + 1, hours_sum, num_features), dtype=np.float32)
    for k in range(len(sample)):
        sample[k] = data[k:k + hours_sum]
    sample_mean = np.nanmean(sample, axis=1)    # Take the mean of the first 6 hours
    return(sample_mean)


def interpol(array):
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(array)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                               (xx, yy),
                               method='cubic')
    return(GD1)


def gender_dummy(demo_ps):
    if demo_ps.shape[0] > 1:
        demo_ps = demo_ps.interpolate()
    demo_ps = demo_ps.interpolate()
    demo_ps = demo_ps.fillna(method='bfill')
    demo_ps = demo_ps.values
    demo_ps = np.insert(demo_ps, -1, demo_ps[:, -1], axis=1)
    demo_ps[:, -1] = abs(demo_ps[:, -1] - 1)
    return(demo_ps)


"""Reaplacing missing values with the mean of the column"""


def process_patient(pati, hours_sum):
    pati_i_mean = summarize_pati(pati, hours_sum)
    col_mean = np.nanmean(pati_i_mean, axis=0)
    inds = np.where(np.isnan(pati_i_mean))
    pati_i_mean[inds] = np.take(col_mean, inds[1])
    return(pati_i_mean)


def get_sepsis_score(data, model):
    """1. Loading data"""
    # Load scaler and imputer for analytics and vital signs --> 0,0037 sec
    imp_an = joblib.load('mean_imputer_an')
    sca_an = joblib.load('scaler_an')
    imp_ps = joblib.load('mean_imputer_ps')
    sca_ps = joblib.load('scaler_ps')

    # Load AE model --> 2 sec
    window = 8
    n_hours = 12

    pred_threshold = 0.0266

    df_demo_ps = np.concatenate((data[:, :7], data[:, 34:35], data[:, 38:40], data[:, 35:36]), axis=1)
    df_an = data[:, 7:34]
    if df_an.shape[0] > 1:
        df_an = process_patient(df_an, n_hours) # 0,0017 sec

    df_demo_ps = pd.DataFrame(data=df_demo_ps)
    df_demo_ps = gender_dummy(df_demo_ps) # 0,022

    # Impute and scale data --> 0,0015

    df_an_mean = imp_an.transform(df_an)
    df_an_mean = sca_an.transform(df_an_mean)
    df_demo_ps = imp_ps.transform(df_demo_ps)
    df_demo_ps = sca_ps.transform(df_demo_ps)

    # Codes
    codes_test = model[1].predict(df_an_mean) # --> 0,167

    # Merging
    data_merged = np.concatenate((df_demo_ps, codes_test), axis=1)

    """Lstm format --> 0,00026"""

    rows_dummy = np.repeat(data_merged, window - 1, axis=0)
    rows_dummy = rows_dummy[0:window - 1, :]

    pre_sample = np.append(rows_dummy, data_merged, axis=0)

    num_rows_current, num_features = pre_sample.shape  # get new shape

    # sliding window
    sample = np.empty((num_rows_current - window + 1, window, num_features), dtype=np.float32)

    for k in range(len(sample)):
        sample[k] = pre_sample[k:k + window]

    ns, nr, nc = sample.shape
    x_test = sample.reshape(ns * nr, nc).reshape(ns, nr * nc)

    prob = model[0].predict(x_test)  # get probabilities --> 1,793
#TODO: Arreglar las primeras muestras que no coinciden con los datos del pychram
    pred = prob.copy()  # prediction array
    pred = pred
    pred[prob >= pred_threshold] = 1  # sepsis
    pred[prob < pred_threshold] = 0  # not sepsis

    return prob[-1], pred[-1]


"""Load models, scalers and imputers"""


def load_sepsis_model():
    model_lstm = load_model('LSTM_July_26_Fold_1.h5', compile=False)
    autoencoder = load_model('AE_July_26_Fold_1.h5')
    encoder = Model(autoencoder.input, autoencoder.layers[-3].output)
    model = {0: model_lstm, 1: encoder}
    return model
