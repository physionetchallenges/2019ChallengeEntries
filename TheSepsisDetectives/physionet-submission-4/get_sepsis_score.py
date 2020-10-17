#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import os, shutil, zipfile
from numpy import array
import csv
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import entropy
import scipy as sc
from zipfile import ZipFile
import joblib

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,Conv2D,MaxPooling2D,Flatten,Conv1D, GlobalMaxPooling1D,MaxPooling1D, Convolution2D,Reshape, InputLayer,LSTM, Embedding
from keras.optimizers import SGD
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from numpy  import array

from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers import TimeDistributed

def load_sepsis_model():
	# Load the saved model pickle file
	Trained_model = joblib.load('saved_model.pkl')
	return Trained_model

def get_sepsis_score(data1, Trained_model):
	#Testing
	t=1
	df_test = np.array([], dtype=np.float64)
	df_test1 = pd.DataFrame()
	l = len(data1)
	
	df_test = data1
	df_test1 = pd.DataFrame(df_test)
	df_test2 = df_test1

	df_test2.columns = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']

	#Forward fill missing values
	df_test2.fillna(method='ffill', axis=0, inplace=True)
	df_test3 = df_test2.fillna(0)
	df_test = df_test3
	
	df_test['ID'] = 0
	DBP = pd.pivot_table(df_test,values='DBP',index='ID',columns='ICULOS')
	O2Sat = pd.pivot_table(df_test,values='O2Sat',index='ID',columns='ICULOS')
	Temp = pd.pivot_table(df_test,values='Temp',index='ID',columns='ICULOS')
	RR = pd.pivot_table(df_test,values='Resp',index='ID',columns='ICULOS')
	BP = pd.pivot_table(df_test,values='SBP',index='ID',columns='ICULOS')
	latest = pd.pivot_table(df_test,values='HR',index='ID',columns='ICULOS')
	
	Fibrinogen = pd.pivot_table(df_test,values='Fibrinogen',index='ID',columns='ICULOS')
	Glucose = pd.pivot_table(df_test,values='Glucose',index='ID',columns='ICULOS')
	HCO3 = pd.pivot_table(df_test,values='HCO3',index='ID',columns='ICULOS')
	WBC = pd.pivot_table(df_test,values='WBC',index='ID',columns='ICULOS')
	HospAdmTime = pd.pivot_table(df_test,values='HospAdmTime',index='ID',columns='ICULOS')
	EtCO2 = pd.pivot_table(df_test,values='EtCO2',index='ID',columns='ICULOS')
	BaseExcess = pd.pivot_table(df_test,values='BaseExcess',index='ID',columns='ICULOS')
	Creatinine = pd.pivot_table(df_test,values='Creatinine',index='ID',columns='ICULOS')
	Platelets = pd.pivot_table(df_test,values='Platelets',index='ID',columns='ICULOS')
	age = pd.pivot_table(df_test,values='Age',index='ID',columns='ICULOS')
	gender = pd.pivot_table(df_test,values='Gender',index='ID',columns='ICULOS')

	Heart_rate_test = latest 
	RR_test = RR 
	BP_test = BP 
	DBP_test = DBP 
	Temp_test = Temp 
	O2Sat_test = O2Sat 

	result = Heart_rate_test

	result = result.fillna(0)
	RR_test = RR_test.fillna(0)
	BP_test = BP_test.fillna(0)
	Temp_test = Temp_test.fillna(0)
	DBP_test = DBP_test.fillna(0)
	O2Sat_test = O2Sat_test.fillna(0)
	
	age = age.fillna(0)
	gender = gender.fillna(0)
	HospAdmTime_test2 = HospAdmTime.fillna(0)
	EtCO2_test2 = EtCO2.fillna(0)
	BaseExcess_test2 = BaseExcess.fillna(0)
	Creatinine_test2 = Creatinine.fillna(0)
	Platelets_test2 = Platelets.fillna(0)
	WBC2_test = WBC.fillna(0)
	HCO32_test = HCO3.fillna(0)
	Glucose2_test = Glucose.fillna(0)
	Fibrinogen2_test = Fibrinogen.fillna(0)
	
	#Since we are using a windows-based approach (6-hour window size), we pad our output for the 6 hours following patients admission.
	#Get dataframe of probs
	#Windows based approach
	Heart_rate_test = result.iloc[:, 0:l]
	RR2_test = RR_test.iloc[:, 0:l]
	BP2_test = BP_test.iloc[:, 0:l]
	Temp2_test = Temp_test.iloc[:, 0:l]
	DBP2_test = DBP_test.iloc[:, 0:l]
	O2Sat2_test = O2Sat_test.iloc[:, 0:l]
	HospAdmTime_test = HospAdmTime_test2.iloc[:, 0:l]
	EtCO22 = EtCO2_test2.iloc[:, 0:l]
	BaseExcess2 = BaseExcess_test2.iloc[:, 0:l]
	Creatinine2 = Creatinine_test2.iloc[:, 0:l]
	Platelets2 = Platelets_test2.iloc[:, 0:l]
	WBC2 = WBC2_test.iloc[:, 0:l]
	gender2 = gender.iloc[:, 0:l]
	HCO32 = HCO32_test.iloc[:, 0:l]
	Glucose2 = Glucose2_test.iloc[:, 0:l]
	Fibrinogen2 = Fibrinogen2_test.iloc[:, 0:l]

	Overall_df_test = pd.concat([Heart_rate_test, BP2_test, Temp2_test, RR2_test, DBP2_test, O2Sat2_test,HospAdmTime_test,EtCO22,BaseExcess2,Creatinine2,Platelets2,gender2,WBC2,HCO32,Glucose2,Fibrinogen2], axis=1)
	Overall_df_test = Overall_df_test.sort_index(axis=1, kind='mergesort')
	Overall_df_test = Overall_df_test.fillna(0)
	
	test_x = Overall_df_test.iloc[:,:].values.reshape(-1, l, 16)
	test_x_norm = test_x
	X_test = test_x_norm
	
	result = Trained_model.predict_proba(X_test) #Gives the probabilities
	
	scores1=result[-1][-1]
	
	if scores1>=0.55:
		labels1 = 1
	else:
		labels1 = 0
				
	return (scores1,labels1)
