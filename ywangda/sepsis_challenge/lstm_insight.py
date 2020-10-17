import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


# data_path = 'D:/Projects/physionet-master/training_data/'
# # raw_df = read_csv(data_path + 'raw_data.csv')
# # full_imputed_df = read_csv(data_path + 'full_imputed_data.csv')
# files = glob.glob(data_path + '*.psv')
# test_dir = 'D:/Projects/physionet-master/test_dir/'
# pred_dir = 'D:/Projects/physionet-master/pred_dir/'
# result_dir = 'D:/Projects/physionet-master/result_dir/'


from pandas import read_csv
from matplotlib import pyplot

'''
def plot_time_series():
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()
'''

'''
# convert series to supervised learning
def series_to_supervised(data, n_time_shift, n_out, dropnan=True):
    n_in = n_time_shift - 1
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # create dataframe
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        # print('drop rows with NaN values')
    return agg
'''

'''
def combine_psv(files, column_names, selected_column_names, column_mean_dict, n_time_shift, scaler, saved_file):
    record_no = np.zeros(len(files))
    file_no = 0

    for file_name in files:
        if file_name.endswith('.psv'):
            (values, _) = read_challenge_data(file_name)
            df = pd.DataFrame(values, columns=column_names)

            # impute missing value by forward and backward filling in or mean
            df.fillna(method='ffill', inplace=True)     # use previous value to fill in
            df.fillna(method='bfill', inplace=True)
            df = df.fillna(value=column_mean_dict)

            # scale the data
            values = df.values
            scaled_data = scaler.fit_transform(values)
            df = pd.DataFrame(scaled_data, columns=column_names)

            # select columns
            df = df.loc[:, selected_column_names]

            # generate series
            feat_df = df.iloc[:, :-1]
            label_df = df.iloc[:, [-1]]
            reframed_df = series_to_supervised(feat_df, n_time_shift, n_out=1, dropnan=True)       # drop first n_time_shift rows
            reframed_df = pd.concat([reframed_df, label_df], axis=1)
            reframed_df.dropna(inplace=True)  # drop rows with NaN values

            # combine the data
            if file_no == 0:
                combined_values = np.array([], dtype=np.float32).reshape(0, len(reframed_df.columns.values))  # +1 denotes the label
                combined_imputed_df = pd.DataFrame(combined_values, columns=reframed_df.columns.values)

            combined_imputed_df = pd.concat([combined_imputed_df, reframed_df], axis=0)
            record_no[file_no] = combined_imputed_df.shape[0]
            file_no += 1
            print(file_no)
            # combined_values = np.vstack([combined_values, values]) if combined_values.size else values

    combined_imputed_df.to_csv(saved_file, index=False)
    print(np.sum(record_no[:4000]))
    print(np.sum(record_no))
'''


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
            vital_feat_df.fillna(mean_basic.to_dict()[0], inplace=True)      # convert dataframe to dict

            # scale the data
            # values = df.values
            # scaled_data = scaler.fit_transform(values)
            # df = pd.DataFrame(scaled_data, columns=column_names)

            # construct the feature
            if feature == 'minimal':
                df = generate_minimal_feature(vital_feat_df, dynamic_column_names, static_column_names, time_win)
            if feature == 'basic':
                df = generate_basic_feature(vital_feat_df, dynamic_column_names, static_column_names, time_win)
            if feature == 'reference':
                df = generate_reference_feature(vital_feat_df, dynamic_column_names, static_column_names, mean_diff, time_win)

            # generate X_test
            values = df.values
            # scaler = joblib.load(scaler_file)
            X_scaled = scaler.fit_transform(values)
            X_test = X_scaled[:, :-1]
            X_test = X_test.reshape((X_test.shape[0], n_time_shift, X_test.shape[1]))

            if df.isnull().values.any():
                print('nan error:', file_name)
                y_prob_imputed = np.zeros((values.shape[0], 1))
                y_pred_imputed = np.zeros((values.shape[0], 1))
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

            print('predicted positive no', np.sum(y_pred))
            print('actual positive no', np.sum(values[:, -1]))
            # write predictions to output file
            output_file = pred_dir + file_name[-10:]
            with open(output_file, 'w') as f:
                f.write('PredictedProbability|PredictedLabel\n')
                for (s, l) in zip(y_prob, y_pred):
                    f.write('%g|%d\n' % (s, l))
    print('total predicted positive no: ', pred_pos)
    print('total actual positive no: ', true_pos)

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
    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], n_time_shift, X_train.shape[1]))
    X_valid = X_valid.reshape((X_valid.shape[0], n_time_shift, X_valid.shape[1]))
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    return X_train, X_valid, y_train, y_valid


def fit_lstm(X_train, X_valid, y_train, y_valid, epoch, batch_size, positive_weight):
    # design network
    model = Sequential()
    # model.add(Conv3D(filters, kernel_size))

    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy', precision_keras, recall_keras])
    # simple early stopping
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=0)
    # fit network
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, class_weight={0.0: 0.5, 1.0: positive_weight}, validation_data=(X_valid, y_valid), verbose=2, shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    return model


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        '''
        train_files = glob.glob('D:/Projects/physionet-master/training_data/training_a/*.psv')
        saved_file = 'D:/Projects/physionet-master/training_data/reduced_lstm_data.csv'
        # column_names = get_columns_names('D:/Projects/physionet-master/training_data/training_a/p00001.psv')
        # print(column_names)
        selected_column_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
        # selected_column_names = column_names
        print(selected_column_names)
        n_time_shift = 3     # t-2, t-1, t
        scaler = joblib.load('scaler.save')
        # combine_psv(train_files, column_names, selected_column_names, column_mean_dict, n_time_shift, scaler, saved_file)
        '''

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
        saved_model = feature + '_lstm.h5'
        test_file = glob.glob('D:/Projects/physionet-master/test_dir/*.psv')
        vital_col_name = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'pH', 'WBC', 'Age', 'HospAdmTime', 'ICULOS',
                          'SepsisLabel']
        dynamic_col_name = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'pH', 'WBC']
        static_col_name = ['Age', 'HospAdmTime', 'ICULOS', 'SepsisLabel']
        n_time_shift = 1
        time_win = 4
        # scaler = joblib.load('scaler.save')


        for ep in [15]:
            for ba in [64]:
                for pos in [1]:    #
                    for pre in [0.5]:
                        epoch = ep
                        batch_size = ba
                        positive_weight = pos    # address imbalanced samples
                        pred_threshold = pre

                        X_train, X_valid, y_train, y_valid = split_data(df, n_time_shift)
                        # model = lstm_model(df, n_time_shift, selected_column_names, epoch, batch_size, positive_weight)
                        model = fit_lstm(X_train, X_valid, y_train, y_valid, epoch, batch_size, positive_weight)
                        predict_psv(model, test_file, feature, vital_col_name, dynamic_col_name,
                                    static_col_name, mean_basic, mean_diff, scaler, pred_threshold, n_time_shift, time_win,
                                     test_dir, pred_dir, result_dir, saved_model)
                        # predict_psv(model, test_files, column_names, selected_column_names, column_mean_dict, pred_threshold, n_time_shift, scaler, test_dir, pred_dir, result_dir, saved_model_file)


