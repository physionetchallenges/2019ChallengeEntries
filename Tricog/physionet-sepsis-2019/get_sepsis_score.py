#!/usr/bin/env python

import numpy as np
import pandas as pd 
import _pickle as cPickle

def get_sepsis_score(patient_data, model):
    rf_loaded_c=model[0];
    rf_loaded_p=model[1];
    pt_data = pd.DataFrame(patient_data)
    pt_data=pt_data.fillna(0)
    testing_pt = np.zeros(6 * pt_data.shape[1])
    WINDOW_SIZE = 6
    # print(WINDOW_SIZE,)
    count = 0
    test_data = []
    # print(len(pt_data[-WINDOW_SIZE:]))
    for row in pt_data[-WINDOW_SIZE:].iterrows():
        index, vitals = row
        vitals = vitals.replace(0, np.mean(vitals))
        vitals = vitals.tolist()
        testing_pt[count:(count+len(vitals))] = vitals
        count += len(vitals)
    # print(count)

    test_data.append(testing_pt)
    classify = rf_loaded_c.predict_proba(test_data)
    predict = rf_loaded_p.predict_proba(test_data)
    scores = []
    labels = []
    row_count = 0
    for prd, clsfy in zip(predict,classify):
        if len(pt_data.index) < 6:
            if np.argmax(clsfy) == 1:
                labels.append(np.argmax(clsfy))
                if np.argmax(clsfy) == 1:
                    scores.append(np.max(clsfy))
                else:
                    scores.append(np.min(clsfy))
            else:
                labels.append(np.argmax(prd))
                if np.argmax(prd) == 1:
                    scores.append(np.max(prd))
                else:
                    scores.append(np.min(prd))   
        else:
            labels.append(np.argmax(prd))
            if np.argmax(prd) == 1:
                scores.append(np.max(prd))
            else:
                scores.append(np.min(prd))

    return scores[-1], labels[-1]


def load_sepsis_model():
    with open('models/rf_classifier.pkl', 'rb') as fid1:
        rf_loaded_c = cPickle.load(fid1)
    with open('models/rf_classifier_pred.pkl', 'rb') as fid2:
        rf_loaded_p = cPickle.load(fid2)
    return [rf_loaded_c, rf_loaded_p]
