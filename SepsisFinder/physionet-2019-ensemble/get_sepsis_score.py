#!/usr/bin/env python3
import sys
import numpy as np
import os, shutil, zipfile
import pandas as pd
from sklearn import ensemble
from joblib import load

from keras.models import Model, load_model
from dataset import PhysionetDatasetCNNInfer

VITALS_COLUMNS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_COLUMNS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']
DEMOGRAPHIC_COLUMNS = ['Age', 'Gender']
HOSPITAL_COLUMNS = ['Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

def load_sepsis_model():
    # cnn_model_filename = "iter_1_ratio_1_2_rand_simple_submitted.h5"
    return {
        "cnn": load_model("iter_0_ratio_1_2_cluster_simpler.h5"),
        "cnn2": load_model("iter_5_ratio_1_2_random_simpler.h5"),
        "rf": load("train_rand_0_5_8_cnn.npz_compressed.joblib"),
        "ensemble": load("ensemble_model_2019_0824_combined.joblib")
        }

def get_sepsis_score(data, model):
    window_size = 24 # TODO: Change to args.window_size?

    threshold = 0.55

    # Assuming that columns will always be in this order hmmmm
    # the fct for loading data only returns numbers no column names so idk
    # df.columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
    data_obj = PhysionetDatasetCNNInfer(data)
    data_obj.__preprocess__(method="measured")
    data_obj.__setwindow__(window_size)

    cnn_model = model["cnn"]
    cnn_model2 = model["cnn2"]
    rf_model = model["rf"]
    ensemble_model = model["ensemble"]

    # CNN prediction
    features = data_obj.__getitem__(data_obj.__len__() - 1)[0]
    X_test = features.reshape(1, window_size, len(data_obj.features), 1)
    y_pred = cnn_model.predict(X_test)
    cnn_score = y_pred.reshape(y_pred.shape[0],)[0]

    y_pred2 = cnn_model2.predict(X_test)
    cnn_score2 = y_pred2.reshape(y_pred2.shape[0],)[0]


    # Random forest prediction
    X_rf = features.reshape(1, window_size * len(data_obj.features))
    rf_score = rf_model.predict_proba(X_rf)[:, 1]

    # Ensemble prediction
    # [["cnn", "cnn2", "rf"]]
    X_ensemble = np.array([cnn_score, cnn_score2, rf_score]).reshape(1, 3)
    ensemble_score = ensemble_model.predict_proba(X_ensemble)[:, 1]

    if ensemble_score >= threshold:
        label = 1
    else:
        label = 0

    # scores = model.predict_proba(test_df_preprocessed)[:, 1]
    # labels = (scores > threshold).astype(int)
    # print(scores)
    # print(labels)

    # This is will only be called for one row at a time so driver.py
    # only expects 1 score and 1 label hmmmmmmmm
    # return scores[0], labels[0]

    return ensemble_score, label
