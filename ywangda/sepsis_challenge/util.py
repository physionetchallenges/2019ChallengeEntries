from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from keras import backend as K


def to_pickle(data, filename):
    with open(filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_scaler(scaler, filename):
    joblib.dump(scaler, filename)


def load_scaler(filename):
    scaler = joblib.load(filename)
    return scaler


def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        col_name = header.split('|')
        value = np.loadtxt(f, delimiter='|')
    '''
    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    '''
    return value, col_name


def split_data(file_name, record_num, train_num, test_num):
    print('loading data...')
    # read the data
    # processed_df = pd.read_csv('D:/Projects/physionet-master/training_data/imputed_scaled_stage1_data.csv')
    processed_df = pd.read_csv(file_name, engine='python')
    # processed_df.loc[586066: 586074, 'iculos_admin_ratio'] = processed_df.loc[586066: 586074, 'iculos_max']/-processed_df.loc[586066: 586074 ,'HospAdmTime']
    column_names = processed_df.columns.values
    feat_names = np.delete(column_names, np.where(column_names == 'SepsisLabel'), axis=0)
    X, y = processed_df[feat_names].values, processed_df['SepsisLabel'].values

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_num, test_size=test_num, shuffle=False)    # do not shuffle
    valid_num = int(0.2 * train_num)
    X_valid, y_valid = X_train[train_num-valid_num:], y_train[train_num-valid_num:]
    X_train, y_train = X_train[:train_num-valid_num], y_train[:train_num-valid_num]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def read_column_name(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        col_name = header.split('|')
    return col_name


def read_raw_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        col_name = header.split('|')
        value = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    '''
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    '''
    return value, col_name

