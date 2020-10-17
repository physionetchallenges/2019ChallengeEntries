#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib  #模型持久化

def get_sepsis_score(data, model):
    data_mean = np.array([
        84.58144, 97.19395, 36.97723, 123.7505, 82.4001,
        63.83056, 18.7265, 32.95766, -0.68992, 24.07548,
        0.554839, 7.378934, 41.02187, 92.65419, 260.2234,
        23.91545, 102.4837, 7.557531, 105.8279, 1.510699,
        1.836177, 136.9323, 2.64666, 2.05145, 3.544238,
        4.135528, 2.114059, 8.290099, 30.79409, 10.43083,
        41.23119, 11.44641, 287.3857, 196.0139, 62.00947,
        0.559269, 0.496571, 0.503429, -56.1251, 26.99499])
    x_std = np.array([
        17.3252421, 2.936924033, 0.770014252, 23.23155568, 16.34175019,
        13.95601, 5.098193617, 7.951662124, 4.294297421, 4.376503698,
        11.12320695, 0.07456771, 9.267241923, 10.89298586, 855.746795,
        19.99431683, 120.1227463, 2.433151846, 5.880461932, 1.805602538,
        3.694082063, 51.31072775, 2.526214355, 0.397897744, 1.423285944,
        0.642149906, 4.311468462, 24.80623535, 5.491748793, 1.968660811,
        26.21766885, 7.731012549, 153.0029077, 103.6353657, 16.38621789,
        0.496474913, 0.499988509, 0.499988509, 162.2569069, 29.00541912])
    data_mean_df = pd.DataFrame(data_mean.reshape(1, 40))
    data = pd.DataFrame(data)
    data = data.fillna(method='pad')
    data = data.fillna(method='bfill')  # 数据病人本身填充

    values = pd.concat([data_mean_df, data], axis=0)
    values = values.fillna(method='pad')  # 引入平均值，再填充

    values.drop(values.columns[[7, 9, 10, 14, 16, 18, 20, 22, 26, 27, 32]], axis=1, inplace=True)
    x_mean = np.delete(data_mean, [7, 9, 10, 14, 16, 18, 20, 22, 26, 27, 32], axis=0)
    x_std = np.delete(x_std, [7, 9, 10, 14, 16, 18, 20, 22, 26, 27, 32], axis=0)

    x_data = values[-1:]  # 将最后一行即当前时间数据作为测试对象
    x_test = (x_data - x_mean) / x_std

    prediction_probas = model.predict_proba(x_test)
    prediction_proba = prediction_probas[-1]
    labels = model.predict(x_test)
    label = labels[-1]

    if label == 1:
        score = max(prediction_proba)
    else:
        score = min(prediction_proba)

    return score, label

def load_sepsis_model():

    model = joblib.load('adaboost_12.pkl')

    return model
