import sys
import numpy as np
import os, shutil, zipfile
import glob
import preprocess_master
import preprocess_median
import preprocess_2500
import pre_combine
import pickle
import copy


def load_sepsis_model():
    return None

est_1=pickle.load(open('finalized_model_1.sav', 'rb'))
est_2=pickle.load(open('finalized_model_2.sav', 'rb'))
est_3=pickle.load(open('finalized_model_3.sav', 'rb'))
est_4=pickle.load(open('finalized_model_4.sav', 'rb'))
est_5=pickle.load(open('finalized_model_5.sav', 'rb'))
est_6=pickle.load(open('finalized_model_6.sav', 'rb'))

est_1_1=pickle.load(open('finalized_model_1.1.sav', 'rb'))
est_2_1=pickle.load(open('finalized_model_2.1.sav', 'rb'))
est_3_1=pickle.load(open('finalized_model_3.1.sav', 'rb'))
est_4_1=pickle.load(open('finalized_model_4.1.sav', 'rb'))
est_5_1=pickle.load(open('finalized_model_5.1.sav', 'rb'))
est_6_1=pickle.load(open('finalized_model_6.1.sav', 'rb'))



def get_sepsis_score(data, model):
    (whole_train,whole_train_median,whole_train_2500)=pre_combine.preprocess(data)
    (matrix_1,matrix_5)=preprocess_master.feature(whole_train)
    (matrix_2,matrix_4)=preprocess_median.feature(whole_train_median)
    (matrix_3,matrix_6)=preprocess_2500.feature(whole_train_2500)

    matrix_1=np.insert(matrix_1,0,0)
    matrix_1=np.insert(matrix_1,0,0)

    matrix_2=np.insert(matrix_2,0,0)
    matrix_2=np.insert(matrix_2,0,0)

    matrix_3=np.insert(matrix_3,0,0)
    matrix_3=np.insert(matrix_3,0,0)

    matrix_4=np.insert(matrix_4,0,0)
    matrix_4=np.insert(matrix_4,0,0)

    matrix_5=np.insert(matrix_5,0,0)
    matrix_5=np.insert(matrix_5,0,0)

    matrix_6=np.insert(matrix_6,0,0)
    matrix_6=np.insert(matrix_6,0,0)

    value_1=est_1.predict(matrix_1.reshape(1,-1))
    value_2=est_2.predict(matrix_2.reshape(1,-1))
    value_3=est_3.predict(matrix_3.reshape(1,-1))
    value_4=est_4.predict(matrix_4.reshape(1,-1))
    value_5=est_5.predict(matrix_5.reshape(1,-1))
    value_6=est_6.predict(matrix_6.reshape(1,-1))
    value_xxx=(value_1+value_2+value_3+value_4+value_5+value_6)/6.0
    
    value_1_1=est_1_1.predict(matrix_1.reshape(1,-1))
    value_2_1=est_2_1.predict(matrix_2.reshape(1,-1))
    value_3_1=est_3_1.predict(matrix_3.reshape(1,-1))
    value_4_1=est_4_1.predict(matrix_4.reshape(1,-1))
    value_5_1=est_5_1.predict(matrix_5.reshape(1,-1))
    value_6_1=est_6_1.predict(matrix_6.reshape(1,-1))
    value_yyy=(value_1_1+value_2_1+value_3_1+value_4_1+value_5_1+value_6_1)/6.0

    value=(value_xxx+value_yyy)/2.0
    




#+value_8+value_9+value_10+value_12+value_13)/12.0k
    #print(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value)
    if (value>1):
        value=1
    if (value<0):
        value=0
    if (value>0.5):
        binary=1
    else:
        binary=0
    return value,binary


