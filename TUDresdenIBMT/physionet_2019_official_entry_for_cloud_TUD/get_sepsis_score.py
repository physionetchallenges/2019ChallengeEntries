#!/usr/bin/env python

# custom imports
import numpy as np
from keras import models
from keras.models import Sequential
from keras.layers import Dense, GRU

def extract_windows(data, parameters, windowLength=20, predictHour=20, extrapolate=True):
    # construct a numpy array (vals) containing the measured values to build the windows
    vals = np.zeros((data.shape[0],len(parameters)))
    for i,par in enumerate(parameters):
        vals[:,i] = data[:,par]
    # fillup the left side of the array with empty values
    if predictHour > 1:
        leftFillUp = np.zeros((predictHour-1,len(parameters)))
        if extrapolate:
            leftFillUp[:,:] = np.nan # fill up with nan values (-> values from the future are unknown for the classifier)
        vals = np.concatenate((leftFillUp,vals),axis=0)
    # fillup the right side of the array with empty values
    if predictHour < windowLength:
        rightFillUp = np.zeros((windowLength-predictHour,len(parameters)))
        if extrapolate:
            rightFillUp[:,:] = vals[-1,:] # fill up with the last known value for each parameter
        vals = np.concatenate((vals,rightFillUp),axis=0)
    # carry forward interpolation for missing values
    for c in range(0,vals.shape[1]): # iterate over each parameter
        lastVal = None
        for r in range(0,vals.shape[0]): # iterate over each row (i.e. parameter value)
            if vals[r,c] != np.nan: # if it is an interpolated value
                lastVal = vals[r,c]
            elif lastVal != None:
                vals[r,c] = lastVal
    # extract the windows
    windows = []
    for i in range(0,data.shape[0]):
        windows.append(vals[i:i+windowLength])
    windows = np.asarray(windows)

    return windows

def build_model_rnn(units, drp=0.2, recdrp=0.2, windowLength=20, parameterQuantity=10):
    model = Sequential()
    for i,u in enumerate(units):
        if i == 0 and len(units) < 2: # if only one recurrent hidden layer
            model.add(GRU(u,
                        dropout=drp,
                        recurrent_dropout=recdrp,
                        input_shape=(windowLength, parameterQuantity)
                        ))
        elif i == 0 and len(units) >= 2: # if more than one recurrent hidden layer return all sequences
            model.add(GRU(u,
                        dropout=drp,
                        recurrent_dropout=recdrp,
                        input_shape=(windowLength, parameterQuantity),
                        return_sequences=True
                        ))
        else: # for the second, third,... layer
            model.add(GRU(u,
                        dropout=drp,
                        recurrent_dropout=recdrp
                        ))
    model.add(Dense(1, activation='sigmoid'))
    return model

def get_sepsis_score(data, rnns):
    # means and standard deviations for input data normalization (derived from expanded dataset ~40000subjects)
    means_train = np.array([84.58144338, 97.19395453, 36.97722824, 123.7504654,
                            63.83055577, 18.72649786, 11.44640502, 62.00946888,
                            7.37893403, 41.02186881])
    stds_train = np.array([17.32523591, 2.93692295, 0.77001352, 23.23154692,
                           13.95600353, 5.09819168, 7.73097368, 16.38621261,
                           0.07456736, 9.26718823])
    # parameters: HR, O2Sat, Temp, SBP, DBP, Resp, WBC, Age, pH, PaCO2
    pars = np.array([0,1,2,3,5,6,31,34,11,12])
    # extract and perprocess the data
    windows = extract_windows(data, pars)
    windows -= means_train
    windows /= stds_train
    windows[np.isnan(windows)] = 0.
    # classify the data
    predicted_scores = [rnns[i].predict(windows).squeeze() for i in range(0,len(rnns))]
    # apply the postprocessing step
    thresholds = [0.4125401266755099, 0.42421763745875213, 0.3991702616907141, 0.3875806318109535]
    predicted_labels = []
    for scores, th in zip(predicted_scores, thresholds):
        if scores.shape == np.zeros(()).shape: # if only one window is classified scores is a float not an array
            scores = np.array((scores,))
        labels = np.zeros(scores.shape)
        applied_th = np.argwhere(scores > th)
        if applied_th.shape[0] > 0:
            first_idx = applied_th[0][0]
        else:
            first_idx = labels.shape[0]
        labels[first_idx:] = 1
        predicted_labels.append(labels)
    # perform an ensemble classification
    for i, labels in enumerate(predicted_labels):
        if i == 0:
            ensemble_labels = labels
        else:
            ensemble_labels += labels
    ensemble_labels[ensemble_labels >= 2] = 1
    # ensemble_labels[ensemble_labels < 2]  = 0
    # adjust the scores for the computation of the utility function
    ensemble_scores = np.zeros(ensemble_labels.shape)
    ensemble_scores[ensemble_labels == 1] = 1.

    return ensemble_scores[-1], ensemble_labels[-1]

def load_sepsis_model():
    # load the model parameters
    folds = ['rnn_fold_1_weights.hdf5', 'rnn_fold_2_weights.hdf5', 'rnn_fold_3_weights.hdf5', 'rnn_fold_4_weights.hdf5']
    rnns = []
    for fold in folds:
        model = build_model_rnn([20])
        model.load_weights(fold)
        rnns.append(model)
    return rnns
