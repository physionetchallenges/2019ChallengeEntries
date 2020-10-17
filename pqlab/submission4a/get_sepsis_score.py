import numpy as np
import pandas as pd
import xgboost as xgb
import os


def deal_data(data):
    # x_mean = [83.8996, 97.052, 36.8055, 126.224, 86.2907, 18.728, 60.8711, 0.5435, -59.6769, 28.4551]
    new_file = np.column_stack((data[:, 0:7], data[:, 34:36], data[:, 38:]))
    nann = pd.isnull(new_file)
    for i in range(len(nann)):
        # 通过平均动脉压=舒张压+1/3*脉压，补缺失的三者中的一个
        if nann[i, 3] or nann[i, 4] or nann[i, 5]:
            if nann[i, 3] & nann[i, 4] == 0 and nann[i, 5]:
                new_file[i, 5] = (new_file[i, 4] - (1 / 3) * new_file[i, 3]) * (3 / 2)
            if nann[i, 5] & nann[i, 4] == 0 and nann[i, 3]:
                new_file[i, 3] = (new_file[i, 4] - (2 / 3) * new_file[i, 5]) * 3
            if nann[i, 5] & nann[i, 3] == 0 and nann[i, 4]:
                new_file[i, 4] = (1 / 3) * new_file[i, 3] - (2 / 3) * new_file[i, 5]
    new_file = np.column_stack((new_file[:, 0:5], new_file[:, 6:]))
    '''
    M = pd.isnull(new_file)
    for i in range(np.size(M, 1)):
        if ~((M[:, i] == 1).all()):
            new_file[:, i] = fill_nan(new_file[:, i])
    '''
    return new_file

    # final_data = to_one(final_data)


def get_sepsis_score(data, model):
    #predict
    n = 5
    label = 0
    score = 0
    bst2 = xgb.Booster({'nthread': 4})  # init model
    bst2.load_model('/physionet2019/xgbB_5.model')
    if len(data) < n:
            score = 0
            label = 0
    if len(data) >= n:
        final_data = deal_data(data)
        temp1 = []
        temp = final_data[-n:, :]
        '''
        for k in range(len(temp)):
            temp1 = np.hstack((temp1, temp[k, 0:6]))
        '''
        temp1 = np.nanmean(temp, 0)
        temp1 = np.hstack((temp1, temp[-1, 6:]))
        temp1 = np.array(temp1).reshape((1, -1))
        temp1 = xgb.DMatrix(temp1)
        score1 = model.predict(temp1)[0]
        score2 = bst2.predict(temp1)[0]
        # y = bst.predict(temp)
        score = (score1 + score2) / 2
        # y = bst.predict(temp)
        thres = 0.45
        if score > thres:
            label = 1
        else:
            label = 0
    return score, label


def load_sepsis_model():
    bst1 = xgb.Booster({'nthread': 4})  # init model
    bst1.load_model('/physionet2019/xgbAA_5.model')
    return bst1
