#!/usr/bin/env python

import xgboost as xgb
import pandas as pd
import numpy as np

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

def get_sepsis_score(data, model):

    # data fill--------------------------------------------------------------------
    pd_data = pd.DataFrame(data)
    pd_temp1 = pd_data.fillna(method='ffill')
    #pd_temp2 = pd_temp1.fillna(value=0)
    pd_temp2 = pd_temp1
    x_train = pd_temp2.values

    # 减6----------------------------------------------------------------
    new_x_train = np.zeros([len(data), 40 + 34 + 7 + 1])  # 40+34+7+1
    Threshold = [100, 22, 2]
    for j in range(34):
        new_index = j * 2

        new_x_train[:, new_index] = x_train[:, j]

        pre_6_reduce = x_train[6:, j] - x_train[:- 6, j]
        new_x_train[6:, new_index + 1] = pre_6_reduce
    new_x_train[:, 68:74] = x_train[:, 34:]

    # ---------------------------增加sepsis---------------------------------------------
    # 收缩压 100 mmhg  呼吸频率22  乳酸 36? 2?(now)
    for select_i, select_v in enumerate([3, 6, 22]):
        temp_data = x_train[:, select_v]
        #index = np.where(temp_data != 0)
        new_x_train[:, 74 + select_i] = temp_data[:] - Threshold[select_i]

    # 血小板 胆红素  肌酐 平均动脉压 mmhg
    for select_i, select_v in enumerate([33, 20, 19, 4]):
        temp_data = x_train[:, select_v]
        #index = np.where(temp_data != 0)
        index = np.where(~np.isnan(temp_data))
        for _, v in enumerate(index[0]):
            flag_val = find_flag(select_v, temp_data[v])
            temp_data[v] = flag_val

        new_x_train[:, 77 + select_i] = temp_data[:]

    #new_x_train[:, 81] = new_x_train[:, 77] + new_x_train[:, 78] + new_x_train[:, 79] + new_x_train[:, 80]
    pd_data1 = pd.DataFrame(new_x_train[:, 77])
    pd_temp1_ = pd_data1.fillna(value=0)
    temp1_ = np.squeeze(pd_temp1_.values)

    pd_data2 = pd.DataFrame(new_x_train[:, 78])
    pd_temp2_ = pd_data2.fillna(value=0)
    temp2_ = np.squeeze(pd_temp2_.values)

    pd_data3 = pd.DataFrame(new_x_train[:, 79])
    pd_temp3_ = pd_data3.fillna(value=0)
    temp3_ = np.squeeze(pd_temp3_.values)

    pd_data4 = pd.DataFrame(new_x_train[:, 80])
    pd_temp4_ = pd_data4.fillna(value=0)
    temp4_ = np.squeeze(pd_temp4_.values)

    new_x_train[:, 81] = temp1_ + temp2_ + temp3_ + temp4_

    dtrain = xgb.DMatrix(new_x_train)
    prediction_resul1 = model[0].predict(dtrain)
    prediction_resul2 = model[1].predict(dtrain)
    prediction_resul3 = model[2].predict(dtrain)
    prediction_resul4 = model[3].predict(dtrain)
    prediction_resul5 = model[4].predict(dtrain)
    prediction_resul = (prediction_resul1+prediction_resul2+prediction_resul3+prediction_resul4+prediction_resul5)/5
    score = prediction_resul[-1]
    if score >0.24:
        label = 1
    else:
        label = 0



    return score, label


def load_sepsis_model():

    xgb1 = xgb.Booster()  # init model
    xgb1.load_model("model/14/5_0")  # load data
    xgb2 = xgb.Booster()  # init model
    xgb2.load_model("model/14/5_1")  # load data
    xgb3 = xgb.Booster()  # init model
    xgb3.load_model("model/14/5_2")  # load data
    xgb4 = xgb.Booster()  # init model
    xgb4.load_model("model/14/5_3")  # load data
    xgb5 = xgb.Booster()  # init model
    xgb5.load_model("model/14/5_4")  # load data


    return [xgb1,xgb2,xgb3,xgb4,xgb5]




