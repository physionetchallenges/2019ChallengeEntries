#!/usr/bin/env python

import numpy as np
import xgboost as xgb
import math

def get_sofa_score(f):
    if f[12]/f[10] < 100: sofa1 = 4
    elif (f[12]/f[10] >= 100) and (f[12]/f[10] < 200): sofa1 = 3
    elif (f[12]/f[10] >= 200) and (f[12]/f[10] < 300): sofa1 = 2
    elif (f[12]/f[10] >= 300) and (f[12]/f[10] < 400): sofa1 = 1
    elif (f[12]/f[10] >= 400): sofa1 = 0
    else: sofa1 = np.nan
    if f[33] < 20: sofa2 = 4
    elif (f[33] >= 20) and (f[33] < 50): sofa2 = 3
    elif (f[33] >= 50) and (f[33] < 100): sofa2 = 2
    elif (f[33] >= 100) and (f[33] < 150): sofa2 = 1
    elif f[33] >= 150: sofa2 = 0
    else: sofa2 = np.nan
    if f[26] < 1.2: sofa3 = 0
    elif (f[26] >= 1.2) and (f[26] < 2): sofa3 = 1
    elif (f[26] >= 2) and (f[26] < 6): sofa3 = 2
    elif (f[26] >= 6) and (f[26] < 12): sofa3 = 3
    elif f[26] >= 12: sofa3 = 4
    else: sofa3 = np.nan
    if f[4] >= 70: sofa4 = 0
    elif f[4] < 70: sofa4 = 1
    else: sofa4 = np.nan
    if f[19] < 1.2: sofa5 = 0
    elif (f[19] >= 1.2) and (f[19] < 2): sofa5 = 1
    elif (f[19] >= 2) and (f[19] < 3.5): sofa5 = 2
    elif (f[19] >= 3.5) and (f[19] < 5): sofa5 = 3
    elif f[19] >= 5: sofa5 = 4
    else: sofa5 = np.nan
    if (f[6] <= 8) or (f[6] >= 25): qsofa1 = 3
    elif (f[6] > 8) and (f[6] <= 11): qsofa1 = 1
    elif (f[6] > 11) and (f[6] <= 20): qsofa1 = 0
    elif (f[6] > 20) and (f[6] <= 25): qsofa1 = 2
    else: qsofa1 = np.nan
    if f[1] <= 91: qsofa2 = 3
    elif (f[1] > 91) and (f[1] <= 93): qsofa2 = 2
    elif (f[1] > 93) and (f[1] <= 95): qsofa2 = 1
    elif f[1] > 95: qsofa2 = 0
    else: qsofa2 = np.nan
    if f[2] <= 35: qsofa3 = 3
    elif (f[2] > 35) and (f[2] <= 36): qsofa3 = 1
    elif (f[2] > 36) and (f[2] <= 38): qsofa3 = 0
    elif (f[2] > 38) and (f[2] <= 39): qsofa3 = 1
    elif f[2] > 39: qsofa3 = 2
    else: qsofa3 = np.nan
    if f[5] <= 90: qsofa4 = 3
    elif (f[5] > 90) and (f[5] <= 100): qsofa4 = 2
    elif (f[5] > 100) and (f[5] <= 110): qsofa4 = 1
    elif (f[5] > 110) and (f[5] <= 219): qsofa4 = 0
    elif f[5] >= 220: qsofa4 = 3
    else: qsofa4 = np.nan
    if f[0] <= 40: qsofa5 = 3
    elif (f[0] > 40) and (f[0] <= 50): qsofa5 = 1
    elif (f[0] > 50) and (f[0] <= 90): qsofa5 = 0
    elif (f[0] > 90) and (f[0] <= 110): qsofa5 = 1
    elif (f[0] > 110) and (f[0] <= 130): qsofa5 = 2
    elif (f[0] > 130): qsofa5 = 3
    else: qsofa5 = np.nan

    return [sofa1, sofa2, sofa3, sofa4, sofa5, qsofa1, qsofa2, qsofa3, qsofa4, qsofa5]

def get_age_quantized(age):
    if age <=20: return 1
    if (age > 20) and (age <= 25): return 2
    if (age > 25) and (age <= 30): return 3
    if (age > 30) and (age <= 35): return 4
    if (age > 35) and (age <= 45): return 5
    if (age > 45) and (age <= 50): return 6
    if (age > 50) and (age <= 55): return 7
    if (age > 55) and (age <= 60): return 8
    if (age > 60) and (age <= 65): return 9
    if (age > 65) and (age <= 70): return 10
    if (age > 70) and (age <= 75): return 11
    if (age > 75) and (age <= 80): return 12
    if (age > 80) and (age <= 85): return 13
    if (age > 85) and (age <= 90): return 14
    else: return 15

def get_sepsis_score(data, model):
    records = []
    filler = [float('nan') for j in range((4)*78)]
    for j,f in enumerate(data):
        for k in range(0, 34):
            if math.isnan(f[k]):
                filler.append(0)
            else:
                filler.append(1)
        sf_score = get_sofa_score(f)
        filler.extend(sf_score)
        f[34] = int(get_age_quantized(f[34]))
        age_vect = np.ndarray.tolist(np.zeros(15))
        age_vect[int(f[34])-1] = 1
        f1 = np.ndarray.tolist(f)
        del f1[34]
        filler.extend(f1[:])
        filler.extend(age_vect)
        records.append(filler[:])
        del filler[:78]
        del filler[len(filler)-20:]
    pred_mat = xgb.DMatrix(np.asarray(records))
    scores = model.predict(pred_mat)
    if scores[-1] >= 0.54:
        pred = 1
    else: pred = 0

    return (scores[-1], pred)
    
def load_sepsis_model():
    bst = xgb.Booster({'nthread': 1})  # init model
    bst.load_model('submission5.model')  # load data
    return bst
