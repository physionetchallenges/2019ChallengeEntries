#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
# import joblib
from sklearn.externals import joblib


def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    return values, column_names


def generate_test_feature(df, dynamic_col_name,  static_col_name, time_win):
    '''
    use pd.shift function
    '''
    # print('generating minimal feature...')
    # print('df shape:', df.shape)

    cols, names = list(), list()
    dynamic_df = df[dynamic_col_name].copy()
    static_df = df[static_col_name].copy()

    if df.shape[0] < time_win:
        for i in range(time_win, 0, -1):
            cols.append(dynamic_df)
            names += [('%s(t-%d)' % (j, i)) for j in dynamic_col_name]
        cols.append(dynamic_df)
        names += [('%s(t)' % j) for j in dynamic_col_name]

        diff_names = list()
        diff_df = dynamic_df.diff(periods=0)
        diff_names += [('%s_diff' % j) for j in dynamic_col_name]
        cols.append(diff_df)
        names += [('%s_diff' % j) for j in dynamic_col_name]

    else:
        for i in range(time_win, 0, -1):
            cols.append(dynamic_df.shift(i))
            names += [('%s(t-%d)' % (j, i)) for j in dynamic_col_name]

        cols.append(dynamic_df)
        names += [('%s(t)' % j) for j in dynamic_col_name]

        diff_names = list()
        diff_df = dynamic_df.diff(periods=time_win)
        # diff_df.fillna(method='ffill', inplace=True)
        # diff_df.fillna(method='bfill', inplace=True)
        diff_names += [('%s_diff' % j) for j in dynamic_col_name]
        cols.append(diff_df)
        names += [('%s_diff' % j) for j in dynamic_col_name]

    cols.append(static_df)
    names += [('%s' % j) for j in static_col_name]

    # concatenate the columns
    minimal = pd.concat(cols, axis=1)
    minimal.columns = names

    # drop nan
    # minimal.dropna(inplace=True)

    # fill nan
    minimal.fillna(method='ffill', inplace=True)
    minimal.fillna(method='bfill', inplace=True)

    # print('minimal feature shape:', minimal.shape)
    # print(minimal.head(5))
    return minimal


def load_sepsis_model():
    model_name = 'util_0.4910_minimal_mlp.h5'
    model = load_model(model_name)                            # save the model
    return model


def refine_label(y_prob, y_pred, consecu_num, time_win):
    n_pred = y_pred.shape[0]
    if n_pred < consecu_num:
        consecu_num = y_pred.shape[0] + time_win

    y_pred_refine = np.zeros((n_pred, 1))
    y_pred_refine[:time_win, 0] = y_pred[time_win, 0]
    y_pred_refine[time_win:, 0] = y_pred[time_win:, 0]

    y_prob_refine = np.zeros((n_pred, 1))
    y_prob_refine[:time_win, 0] = y_prob[time_win, 0]
    y_prob_refine[time_win:, 0] = y_prob[time_win:, 0]

    i = n_pred
    while i > consecu_num:
        if np.sum(y_pred_refine[i-consecu_num:i, 0]) == consecu_num:
            y_pred_refine[i-consecu_num:i, 0] = 1
        else:
            y_pred_refine[i-consecu_num:i, 0] = 0
            y_prob_refine[i-consecu_num:i, 0] = 0
        i -= 1
    return y_prob_refine, y_pred_refine


def get_sepsis_score(data, model):
    # print('record:', data.shape[0])
    cont_label_num = 3
    time_win = 4         # t-4, t-3, t-2, t-1, t
    pred_threshold = 0.5
    col_name = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
                'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
                'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
                'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
    vital_col_name = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'pH', 'WBC', 'Age', 'HospAdmTime', 'ICULOS']
    dynamic_col_name = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'pH', 'WBC']
    static_col_name = ['Age', 'HospAdmTime', 'ICULOS']

    if data.shape[0] < time_win + 1:
        # print('smaller than time_win')
        return 0, 0
    # scaler_name = cwd + '\minmax_scaler.save'
    scaler = joblib.load('minmax_scaler.save')
    f = open('mean_basic', 'rb')
    mean_basic = pickle.load(f)

    df = pd.DataFrame(data, columns=col_name)
    # print('df shape:', df.shape)
    vital_feat_df = df[vital_col_name].copy()  # select vital feature
    # print('vital_feat_df shape:', vital_feat_df.shape)

    # impute missing value by forward, backward and median filling in
    vital_feat_df.fillna(method='ffill', inplace=True)
    vital_feat_df.fillna(method='bfill', inplace=True)
    vital_feat_df.fillna(mean_basic.to_dict()[0], inplace=True)  # convert dataframe to dict

    if vital_feat_df.isnull().values.any():
        raise Exception('nan error')

    # vital_feat_df.fillna(mean_diff.to_dict()[0], inplace=True)  # convert dataframe to dict
    # construct the feature
    feat_df = generate_test_feature(vital_feat_df, dynamic_col_name, static_col_name, time_win)

    # generate X_test
    values = feat_df.values
    # print(':', values.shape)
    X_scaled = scaler.transform(values)
    # X_test = X_scaled[:, :-1]  # remove label
    # X_test = X_test.reshape((X_test.shape[0], n_time_shift, X_test.shape[1]))

    scores = model.predict(X_scaled)  # probability
    labels = scores > pred_threshold  # key parameter
    scores, labels = refine_label(scores, labels, cont_label_num, time_win)
    return scores[-1], labels[-1]




