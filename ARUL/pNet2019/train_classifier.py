#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code tested on python version 3.7.3

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import glob


if __name__ == '__main__':
    
    trainingData_directory = '/mitoC/physionet/inputFiles/training_setB/' # trained on training_setB files
    trainingData_fileList = glob.glob(os.path.join(trainingData_directory + '*.psv'))
    dfList = []
    print("In progress: Reading training data")
    for inputfile in trainingData_fileList:
        single_df=pd.read_csv(inputfile,sep="|",index_col=None,header=0)
        dfList.append(single_df)
    data = pd.concat(dfList, axis=0, ignore_index=True) # Concatenate every patient file into single training dataset
    print("In progress: Pre-processing training data")
    #print data.shape
    #print data.tail()
    data.dropna(axis='rows', thresh=10, inplace=True)
    #print data.shape
    #print data.describe()
    data.fillna(data.mean(),inplace=True)
    #print data.isnull().sum()
    #print data.shape
    print("In progress: Training model")
    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'SepsisLabel'], data['SepsisLabel'], stratify=data['SepsisLabel'], test_size=0.15, random_state=66)
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))
    filename = 'my_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    f = open('trainingData_colMeans.txt','w')
    data_colMean = data.mean(axis=0)
    for i in range(0, len(data_colMean)):
        f.write(str(data_colMean[i]) + '\n')
    f.close()
    print("Completed: Successfully trained model - my_model.pkl")
