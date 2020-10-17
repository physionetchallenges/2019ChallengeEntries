#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on 2019/05/10
@author: hezhengling15@mails.ucas.ac.cn
"""  
from keras.models import Model
from sklearn.externals import joblib
from hzlCommon import *
from keras.preprocessing import *
from keras import backend as K 
from keras.models import load_model
import numpy as np 
myC=None
binCodes=[[0,1,1,0],[0,1,0,1],[1,1,0,0],[0,1,1,1],[1,1,1,0]]
modelNames=["XGB","GBT","XGB2"] 
N=5
dataPool=np.zeros(N)
poolIndex=0
def costLoss(y_true, y_pred): 
#     y_true = K.round(y_input) 
    print("shape of y_true",K.shape(y_true))
    c_FN = 2
    c_TP=0
    c_FP=1
    c_TN=0
    cost = y_true * K.log(y_pred) *c_FN + y_true * K.log(1 - y_pred)* c_TP+(1 - y_true) * K.log(1 -y_pred) * c_FP + (1 - y_true)*K.log(y_pred) * c_TN
    return - K.mean(cost, axis=-1)
def sensitivity(y_true, y_pred,isOneHot=False):
    if isOneHot:#注意，如果是one-hot编码，需要处理一下
        y_true=K.argmax(y_true,axis=1)
        y_pred=K.argmax(y_pred,axis=1)   
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives   )
def specificity(y_true, y_pred,isOneHot=False):
    if isOneHot:#注意，如果是one-hot编码，需要处理一下
        y_true=K.argmax(y_true,axis=1)
        y_pred=K.argmax(y_pred,axis=1)  
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives   ) 
def miniFix(predict_posibility):
    global N,dataPool,poolIndex
    dataPool[poolIndex%N]=predict_posibility
    poolIndex+=1
    return np.average(dataPool)

def constructLSTM_XGB_Features(train_x,lstm4Models,lstm5Models,lstm6Models):
    new_data_x=[]
    rawFeatureLen=40
    for i in range(len(train_x)):  
        d=train_x[i]
        tmpFeature=[]
        for m in lstm4Models:
#             tmpD=m.predict(np.array([d[2:,:rawFeatureLen]]))[0]
#             print("np.array(tmpD).shape-----",np.array(tmpD).shape)
            tmpFeature+=m.predict(np.array([d[2:,:rawFeatureLen]]))[0].tolist()
        for m in lstm5Models:
            tmpFeature+=m.predict(np.array([d[1:,:rawFeatureLen]]))[0].tolist()
        for m in lstm6Models:
            tmpFeature+=m.predict(np.array([d[0:,:rawFeatureLen]]))[0].tolist()
# #         print(np.array(tmpFeature).shape,np.array(d[-1]).shape)
#         tmpFeature+=d[-1].tolist()
        tmpFeature+=d[-1].tolist()
        new_data_x.append([tmpFeature])  
    return np.array(new_data_x)

def votingMethod1(predictions): 
    weight=[24.0,24.0,6.0,24.0,6.0,16.0,16.0,4.0,16.0,4.0,19.2,19.2,4.8,19.2,4.8]
    #weight=[16.0,16.0,4.0,16.0,4.0,24.0,24.0,6.0,24.0,6.0,19.2,19.2,4.8,19.2,4.8]
    #weight=[16.0,16.0,4.0,16.0,4.0,24.0,24.0,6.0,24.0,6.0,12.8,12.8,3.2,12.8,3.2,19.2,19.2,4.8,19.2,4.8]
    pos=np.sum(np.array(predictions)*np.array(weight))/np.sum(weight)
    if pos<0.001:
        pos=0.001
    if pos>0.999:
        pos=0.999
    return pos
def votingMethod2(predictions): 
    #print(predictions) 
    return np.sum(np.array(predictions)>0.45)/len(predictions)

def getSaveModelName(mdName,binCode,version):
    s=""
    for b in binCode:
        s+=str(b)
    return mdName+"_"+s+"_"+str(version)
def getSelections(binCode):
#     binCode: RAW+SOFA=[0,1,1,0];RAW+RIRS=[0,1,0,1];RAW+LSTM=[1,1,0,0]
#              RAW+SOFA+SIRS=[0,1,1,1];RAW+SOFA+LSTM=[1,1,1,0];RAW+SOFA+SIRS+LSTM=[1,1,1,1]

    #LSTM,RAW,SOFA,SIRS, the size of each segment is [64,40,17,13]
    sgLen=[12,40,17,13] #the size of each segment
    if len(sgLen)!=4 or len(binCode)!=4:
        print("Error!!! The length of sgLen or binCode should be 4.")
        return
    sumOfsgLen=[]
    s=0
    for i in range(len(sgLen)):
        s+=sgLen[i]
        sumOfsgLen.append(s)
        
    res=[]
    if binCode[0]==1:
        a=[i for i in range(sumOfsgLen[0])]
        res+=a
    if binCode[1]==1:
        a=[i for i in range(sumOfsgLen[0],sumOfsgLen[1])]
        res+=a
    if binCode[2]==1:
        a=[i for i in range(sumOfsgLen[1],sumOfsgLen[2])]
        res+=a
    if binCode[3]==1:
        a=[i for i in range(sumOfsgLen[2],sumOfsgLen[3])]
        res+=a
    return res


def get_sepsis_score(values, model):   
    global myC,binCodes,modelNames
    timestamps=6
    curLen=len(values)
    s=curLen-timestamps
    if s<0:
        s=0
    uX=myC.constructFeaturesAndNormalize(values,s,curLen)    
    newValues=sequence.pad_sequences(np.array([uX]), maxlen=timestamps,dtype='float', value=-1,padding='pre')
    

    
    new_train_x=constructLSTM_XGB_Features(newValues,model[0],model[1],model[2])
    #print(np.array(new_train_x).shape)
    predictions=[]
    curIndex=0
    for t in modelNames:
        for bC in binCodes:
            predVector=np.array(new_train_x[0])
            curPosible=model[-1][curIndex].predict(predVector[:,getSelections(bC)])[0]
            predictions.append(curPosible)
            curIndex+=1
            
#    predict_posibility=miniFix(model[-1].predict(new_train_x[0]))        
    predict_posibility=votingMethod1(predictions)
    if predict_posibility>0.45: 
        predict_label=1
    else:
        predict_label=0 
                 
    return predict_posibility, predict_label
def load_sepsis_model():
    global myC,binCodes,modelNames
    featureOutLayer=-2
    myC=hzlCommon()

    
    m42 = load_model("./model_single_v4_2_0.h5",custom_objects={'sensitivity': sensitivity,'specificity': specificity}) 


    # m50 = load_model("./hzlmodel/model_single_v5_0_0.h5",custom_objects={'sensitivity': sensitivity,'specificity': specificity}) 
    m51 = load_model("./model_single_v5_1_0.h5",custom_objects={'sensitivity': sensitivity,'specificity': specificity}) 
    # m52 = load_model("./hzlmodel/model_single_v5_2_0.h5",custom_objects={'sensitivity': sensitivity,'specificity': specificity}) 

    m60 = load_model("./model_single_v6_0_0.h5",custom_objects={'sensitivity': sensitivity,'specificity': specificity}) 
    # m61 = load_model("./hzlmodel/model_single_v6_1_0.h5",custom_objects={'sensitivity': sensitivity,'specificity': specificity}) 
    # m62 = load_model("./hzlmodel/model_single_v6_2_0.h5",custom_objects={'sensitivity': sensitivity,'specificity': specificity}) 
    lstm4Models=[m42]
    lstm5Models=[m51]
    lstm6Models=[m60]
    
    
    ensembleModels=[]
    for t in modelNames:
        for bC in binCodes:
            tmpName=getSaveModelName(t,bC,0)
            tmpName='./model_'+tmpName+'.pkl'
            ensembleModels.append(joblib.load(tmpName))
    
    lstm4Models=[Model(inputs=m.input, outputs=m.layers[featureOutLayer].output) for m in lstm4Models]
    lstm5Models=[Model(inputs=m.input, outputs=m.layers[featureOutLayer].output) for m in lstm5Models]
    lstm6Models=[Model(inputs=m.input, outputs=m.layers[featureOutLayer].output) for m in lstm6Models]
    
    
    model=[lstm4Models,lstm5Models,lstm6Models,ensembleModels]
    return model
