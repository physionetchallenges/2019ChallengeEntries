import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import lightgbm as lgb
from math import sqrt
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
# from pandas import DataFrame
# from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Conv3D
from keras.callbacks import EarlyStopping
from sklearn.externals import joblib
from compute_scores_2019 import compute_accuracy_f_measure, compute_prediction_utility, compute_auc, compute_scores_2019
import tensorflow as tf

from datetime import datetime
import glob
# load data
import pandas as pd
from util import *
from preprocess_insight import *
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import read_csv
from matplotlib import pyplot


def precision_keras(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_keras(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    precision = precision_keras(y_true, y_pred)
    recall = recall_keras(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def predict_psv(model, test_file, feature, vital_column_names, dynamic_column_names, static_column_names, mean_basic,
                mean_diff, scaler, pred_threshold, n_time_shift, time_win, test_dir, pred_dir, result_dir,
                saved_model_file):
    pred_pos = 0
    true_pos = 0
    file_no = 0
    for file_name in test_file:
        if file_name.endswith('.psv'):
            file_no += 1
            print('---------------------------------------------------------------------------------------------------')
            print('predicting ', file_name, ' process: ', file_no, '/', len(test_file))
            (values, col_name) = read_raw_data(file_name)
            df = pd.DataFrame(values, columns=col_name)
            n_pred = df.shape[0]
            vital_feat_df = df[vital_column_names]                 # select vital feature

            # impute missing value by forward, backward and median filling in
            vital_feat_df.fillna(method='ffill', inplace=True)
            vital_feat_df.fillna(method='bfill', inplace=True)
            # vital_feat_df.fillna(mean_basic.to_dict()[0], inplace=True)      # convert dataframe to dict
            # vital_feat_df.fillna(mean_diff.to_dict()[0], inplace=True)
            vital_feat_df.fillna(mean_basic, inplace=True)

            # construct the feature
            if feature == 'minimal':
                df = generate_minimal_feature(vital_feat_df, dynamic_column_names, static_column_names, time_win)
            if feature == 'basic':
                df = generate_basic_feature(vital_feat_df, dynamic_column_names, static_column_names, time_win)
            if feature == 'reference':
                df = generate_reference_feature(vital_feat_df, dynamic_column_names, static_column_names, mean_diff, time_win)

            # generate X_test
            values = df.values
            X_scaled = scaler.fit_transform(values)
            X_test = X_scaled[:, :-1]                # remove label
            y_true = X_scaled[:, -1]

            if df.isnull().values.any():
                print('nan error:', file_name)
                y_prob = np.zeros((values.shape[0], 1))
                y_pred = np.zeros((values.shape[0], 1))
            else:
                y_prob = model.predict(X_test)         # probability
                y_pred = y_prob > pred_threshold       # key parameter
                if y_prob.shape[0] != n_pred:
                    raise Exception()
                # y_prob_imputed = np.zeros((n_pred, 1))       # the record num decrease after reframing
                # y_prob_imputed[n_time_shift-1:, 0] = y_prob[:, 0]
                # for i in range(n_time_shift-1):
                #     y_prob_imputed[i, 0] = y_prob_imputed[n_time_shift-1, 0]
                # y_pred_imputed = np.zeros((n_pred, 1))
                # y_pred_imputed[n_time_shift-1:, 0] = y_pred[:, 0]
                # for i in range(n_time_shift-1):
                #     y_pred_imputed[i, 0] = y_pred_imputed[n_time_shift-1, 0]

            pred_pos += np.sum(y_pred)
            true_pos += np.sum(values[:, -1])

            print('predicted positive no:', np.sum(y_pred))
            print('actual positive no:', np.sum(values[:, -1]))
            # write predictions to output file
            output_file = pred_dir + file_name[-10:]
            with open(output_file, 'w') as f:
                f.write('PredictedProbability|PredictedLabel\n')
                for (s, l) in zip(y_prob, y_pred):
                    f.write('%g|%d\n' % (s, l))
    print('total predicted positive no:', pred_pos)
    print('total actual positive no:', true_pos)

    # compute score
    auroc, auprc, accuracy, f_measure, utility = compute_scores_2019(test_dir, pred_dir)
    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}\n'.format(auroc, auprc, accuracy, f_measure,
                                                                                      utility)
    print(output_string)

    # save model weights
    model.save(result_dir + 'util_' + str(utility) + '_' + saved_model_file)  # save the model

    # save results to file
    result_file = result_dir + 'util_' + str(utility) + '.txt'
    if result_file:
        with open(result_file, 'w+') as f:
            f.write(output_string)


def split_data(df, n_time_shift):
    # split into train and test sets
    values = df.values
    n_train = int(0.8*values.shape[0])
    train = values[:n_train, :]
    test = values[n_train:, :]
    # split into input and outputs
    X_train, y_train = train[:, :-1], train[:, -1]
    X_valid, y_valid = test[:, :-1], test[:, -1]
    print('data shape: ', X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    return X_train, X_valid, y_train, y_valid


def fit_lgb(X_train, X_valid, y_train, y_valid, pos_weight):
    print('training by sk lgb...')
    gbm = lgb.LGBMClassifier(objective='binary',
                             learning_rate=0.1,
                             n_estimators=100,       # 100
                             metric='binary_logloss',
                             num_leaves=71,
                             max_depth=5,            # 6
                             subsample_for_bin=200000,
                             class_weight={0: 0.5, 1: pos_weight},
                             min_split_gain=0.0,
                             min_child_weight=0.001,
                             min_child_samples=30,
                             subsample=0.6,
                             subsample_freq=0,
                             colsample_bytree=0.8,
                             reg_alpha=0.01,
                             reg_lambda=0.001,
                             random_state=None,
                             n_jobs=4,
                             silent=True)

    # train
    gbm.fit(X_train, y_train)
    return gbm


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        test_dir = 'D:/Projects/physionet-master/test_dir/'
        pred_dir = 'D:/Projects/physionet-master/pred_dir/'
        result_dir = 'D:/Projects/physionet-master/result_dir/'
        feature = 'minimal'
        scaler_name = "scaler.save"
        scaler = joblib.load(scaler_name)
        f = open('mean_basic', 'rb')
        mean_basic = pickle.load(f)
        # print(mean_basic)
        f = open('mean_diff', 'rb')
        mean_diff = pickle.load(f)
        # print(mean_diff)

        if feature == 'minimal':
            file = 'D:/Projects/physionet-master/training_data/minimal_df.csv'
        if feature == 'basic':
            file = 'D:/Projects/physionet-master/training_data/basic_df.csv'
        if feature == 'reference':
            file = 'D:/Projects/physionet-master/training_data/reference_df.csv'

        df = read_csv(file)
        saved_model = feature + '_lgb.h5'
        test_file = glob.glob('D:/Projects/physionet-master/test_dir/*.psv')
        vital_col_name = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'pH', 'WBC', 'Age', 'HospAdmTime', 'ICULOS',
                          'SepsisLabel']
        dynamic_col_name = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'pH', 'WBC']
        static_col_name = ['Age', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
        n_time_shift = 1
        time_win = 4
        # scaler = joblib.load('scaler.save')
        pred_threshold = 0.5
        pos_weight = 30

        X_train, X_valid, y_train, y_valid = split_data(df, n_time_shift)
        a = np.sum(y_train)
        b = np.sum(y_valid)
        model = fit_lgb(X_train, X_valid, y_train, y_valid, pos_weight)
        predict_psv(model, test_file, feature, vital_col_name, dynamic_col_name,
                    static_col_name, mean_basic, mean_diff, scaler, pred_threshold, n_time_shift, time_win,
                     test_dir, pred_dir, result_dir, saved_model)
