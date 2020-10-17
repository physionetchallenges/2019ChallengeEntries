# -*- coding: utf-8 -*-
# @Time    : 2019/7/8 21:07
# @Author  : LVM
# @FileName: get_sepsis_score.py
# @Software: PyCharm
# @Blog    ：
'''゜○。○ｏ°゜゜ｏ○。ｏ゜○。゜Ｏ○。°゜Ｏ○。°○ｏ°○ｏ○゜ｏ。Ｏ゜○。゜。゜○。○゜。Ｏ°ｏ○。ｏ゜゜○。○゜。
Ｏ°ｏ○。ｏ○○○゜Ｏ ○。°゜゜。○○○ｏ°゜ｏ○○○○゜゜。゜○○○○゜○。○゜○Ｏ○ｏ○。ｏ゜゜○。○゜。Ｏ°ｏ○。ｏ゜
○。゜Ｏ○。○   ○゜。°○ｏ°○ｏ○   ○。Ｏ゜○     ○゜゜。○        ○。ｏ゜○       ○°ｏ○。ｏ゜○。゜Ｏ○。°ｏ゜○
゜Ｏ○。゜。○   ○ｏ○゜ｏ。Ｏ゜○。○    ○゜○   Ｏ°ｏ○ ○    ○○    ○ｏ○    ○○   ○゜。°゜Ｏ○。°○ｏｏ゜○
°○ｏ○゜ｏ○   ○○○○。゜○。○゜。○    ○   ○゜゜○。○    ○°○     ○    ○゜○    ○゜。°○ｏ°○ｏ○゜ｏ。
Ｏ゜○。゜。○ L        ○。ｏ ○。○゜。○  V  ○。ｏ゜。○    ○。 Ｏ○    M    ○゜ｏ○    ○゜○。゜。゜ｏ゜○ｏ゜○
○。○゜。Ｏ○○○○○○゜○。○゜。Ｏ°ｏ ○○○゜○○゜Ｏ○○○゜Ｏ○。°○ ○ ○ｏ○゜○ ○゜○。゜。゜○。○゜。Ｏ°
ｏ○。ｏ゜゜○。○゜。Ｏ°ｏ○。ｏ゜○。゜Ｏ○。○゜Ｏ○。°○ｏ°○ｏ○゜ｏ。Ｏ゜○。゜。゜○。○゜。Ｏ°ｏ○。ｏ゜゜'''
import os
import sys

import numpy as np
import pandas as pd
from sklearn.externals import joblib


def add_lag_features(values):
    if len(values)>6:
        features_lag = []
        lag_features = values[:-6, :35] - values[6:, :35]
        lag_features_Temp = lag_features[:,2]
        lag_features_Fio2 = lag_features[:,10]
        lag_features_Lacate = lag_features[:,22]
        lag_features_WBC = lag_features[:,31]

        features_lag.append(lag_features_Temp)
        features_lag.append(lag_features_Fio2)
        features_lag.append(lag_features_Lacate)
        features_lag.append(lag_features_WBC)
        features_lag = np.array(features_lag)
        features_lag = features_lag.transpose((1,0))
        features_lag = features_lag.tolist()
        training_example = np.append(values[-1], features_lag[-1])
    else:
        pad = np.zeros((1, 4))
        training_example = np.append(values[-1], pad)
    return training_example

def load_sepsis_model():
    models_root = './logs/2019-08-20-16-01-03'#2019-08-19-11-58-16  2019-07-23-23-58-45 2019-07-22-22-45-34 2019-07-08-20-38-16 '../data/logs/2019-07-08-20-38-16'
    models_paths = [os.path.join(models_root, m) for m in os.listdir(models_root) if m.endswith('.bin')]
    models_paths.sort()
    model0 = joblib.load(models_paths[0])
    model1 = joblib.load(models_paths[1])
    model2 = joblib.load(models_paths[2])
    model3 = joblib.load(models_paths[3])
    model4 = joblib.load(models_paths[4])

    return [model0,model1,model2,model3,model4]

def find_flag(flag_index, val):
    if flag_index == 33: # 血小板
        if val <= 150 and val>100:
            return 1
        elif val<=100 and val>50:
            return 2
        elif val<=50 and val>20:
            return 3
        elif val<=20:
            return 4
        else:
            return 0
    if flag_index == 20: # 胆红素
        if val>=1.2 and val <1.9:
            return 1
        elif val>=1.9 and val<5.9:
            return 2
        elif val>=5.9 and val<11.9:
            return 3
        elif val>=11.9:
            return 4
        else:
            return 0
    if flag_index == 19: # 肌酐
        if val >=1.2 and val<1.9:
            return 1
        elif val>=1.9 and val<3.4:
            return 2
        elif val>=3.4 and val<4.9:
            return 3
        elif val>=4.9:
            return 4
        else:
            return 0
    if flag_index == 4:  # 平均动脉压
        if val < 70:
            return 1
        else:
            return 0
def get_sepsis_score(values,model):

    # generate predictions
    thr = 0.29
    example = pd.DataFrame(values, columns=None, copy=True)
    example.ffill(inplace=True)
    #example.bfill(inplace=True)
    example.fillna(0, inplace=True)
    #model0,model1,model2,model3,model4 = model()
    #scores = np.zeros((1,))
    #new_x_train = np.zeros([len(y_train), 40 + 34 + 7 + 1])  # 40+34+7+1
    new_x_train = np.zeros((len(values), 82))
    Threshold = [100, 22, 2]
    x_train = example.values
    for j in range(34):
        new_index = j * 2

        new_x_train[:, new_index] = x_train[:, j]
        if len(x_train)>0:
            pre_6_reduce = x_train[6:, j] - x_train[:- 6, j]
            new_x_train[6:, new_index + 1] = pre_6_reduce
        # else:
        #     pre_6_reduce = np.zeros((len(x_train),1))
        #     new_x_train[:6, new_index + 1] = pre_6_reduce
    new_x_train[:, 68:74] = x_train[:, 34:]

    # ---------------------------增加sepsis---------------------------------------------
    # 收缩压 100 mmhg  呼吸频率22  乳酸 36? 2?(now)
    for select_i, select_v in enumerate([3, 6, 22]):
        temp_data = x_train[:, select_v]
        index = np.where(temp_data != 0)
        new_x_train[index, 74 + select_i] = temp_data[index] - Threshold[select_i]

    # 血小板 胆红素  肌酐 平均动脉压 mmhg
    for select_i, select_v in enumerate([33, 20, 19, 4]):
        temp_data = x_train[:, select_v]
        index = np.where(temp_data != 0)
        for _, v in enumerate(index[0]):
            flag_val = find_flag(select_v, temp_data[v])
            temp_data[v] = flag_val

            new_x_train[index, 77 + select_i] = temp_data[index]

        new_x_train[:, 81] = new_x_train[:, 77] + new_x_train[:, 78] + new_x_train[:, 79] + new_x_train[:, 80]

    #Add_example = add_lag_features(example.values)
    #Add_example = Add_example.reshape(1, -1)
    example = new_x_train
    scores1= model[0].predict_proba(example)[:, 1]
    scores2 = model[1].predict_proba(example)[:, 1]
    scores3= model[2].predict_proba(example)[:, 1]
    scores4= model[3].predict_proba(example)[:, 1]
    scores5= model[4].predict_proba(example)[:, 1]
    scores = (scores1 + scores2 + scores3 + scores4 + scores5) / 5
    # if(len(example.values)>1):
    #     scores = (scores1+scores2+scores3+scores4+scores5)/5
    # else:
    #     scores=(scores1+scores2+scores3+scores4+scores5)/50
        #scores += m.predict_proba(values)[:, 1]
    labels = np.where(scores > thr, 1, 0)
    return scores[-1],labels[-1]

