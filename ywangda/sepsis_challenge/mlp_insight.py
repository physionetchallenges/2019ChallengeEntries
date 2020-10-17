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
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier

# data_path = 'D:/Projects/physionet-master/training_data/'
# # raw_df = read_csv(data_path + 'raw_data.csv')
# # full_imputed_df = read_csv(data_path + 'full_imputed_data.csv')
# files = glob.glob(data_path + '*.psv')
# test_dir = 'D:/Projects/physionet-master/test_dir/'
# pred_dir = 'D:/Projects/physionet-master/pred_dir/'
# result_dir = 'D:/Projects/physionet-master/result_dir/'


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
            X_test = X_scaled[:, :-1]                # remove label
            # X_test = X_test.reshape((X_test.shape[0], n_time_shift, X_test.shape[1]))

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
    # reshape input to be 3D [samples, timesteps, features]
    #X_train = X_train.reshape((X_train.shape[0], n_time_shift, X_train.shape[1]))
    #X_valid = X_valid.reshape((X_valid.shape[0], n_time_shift, X_valid.shape[1]))
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    return X_train, X_valid, y_train, y_valid


def fit_mlp(X_train, X_valid, y_train, y_valid, epoch, batch_size, pos_weight):
    #pos_weight = 30
    ker_init = 'he_normal'
    # batch_size = 16
    activation = 'relu'
    hidden_dim = 64
    hidden_dim2 = 8
    drop_out = 0.1
    prob_thresh = 0.5
    run = 1

    # undropped_columns = [x for x in list(column_names) if x not in dropped_columns]

    # print('train positive:', int(np.sum(y_train)))
    # print('Test positive:', int(np.sum(y_test)))

    input_dim = X_train.shape[1]
    from sklearn.utils import class_weight

    # build neural network
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.5))

    model.add(Dense(100, input_dim=200, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.25))

    model.add(Dense(50, input_dim=100, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.1))

    model.add(Dense(25, input_dim=50, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.05))

    model.add(Dense(1, input_dim=25, kernel_initializer=ker_init))
    model.add(Activation('sigmoid'))

    # optimization setting
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[recall_keras, precision_keras, f1])

    # compute class weight
    # class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    # print('class_weight:', {0.0: 0.5, 1.0: pos_weight})
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    # ModelCheckpoint(filepath=result_dir + 'pos_weight_' + str(pos_weight) + '_mlp.h5', monitor='val_loss', save_best_only=True)]
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)  # chekc localhost:6006

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1, shuffle=True,
              class_weight={0.0: 0.5, 1.0: pos_weight}, validation_data=(X_valid, y_valid),
              callbacks=[early_stop])

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    return model


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
        f = open('mean_diff', 'rb')
        mean_diff = pickle.load(f)

        if feature == 'minimal':
            file = 'D:/Projects/physionet-master/training_data/minimal_df.csv'
        if feature == 'basic':
            file = 'D:/Projects/physionet-master/training_data/basic_df.csv'
        if feature == 'reference':
            file = 'D:/Projects/physionet-master/training_data/reference_df.csv'

        df = read_csv(file)
        saved_model = feature + '_mlp.h5'
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
                for pos in [25]:                 #
                    for pre in [0.5]:
                        epoch = ep
                        batch_size = ba
                        positive_weight = pos    # address imbalanced samples
                        pred_threshold = pre

                        X_train, X_valid, y_train, y_valid = split_data(df, n_time_shift)
                        # model = lstm_model(df, n_time_shift, selected_column_names, epoch, batch_size, positive_weight)
                        model = fit_mlp(X_train, X_valid, y_train, y_valid, epoch, batch_size, positive_weight)
                        predict_psv(model, test_file, feature, vital_col_name, dynamic_col_name,
                                    static_col_name, mean_basic, mean_diff, scaler, pred_threshold, n_time_shift, time_win,
                                     test_dir, pred_dir, result_dir, saved_model)
                        # predict_psv(model, test_files, column_names, selected_column_names, column_mean_dict, pred_threshold, n_time_shift, scaler, test_dir, pred_dir, result_dir, saved_model_file)


