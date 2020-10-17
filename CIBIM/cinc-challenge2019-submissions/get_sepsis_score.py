#!/usr/bin/env python

import numpy as np

def get_sepsis_score(features, all_model):
    
    feat_min=[20,20,20,20,20,20,1.5,10,-40,5,0.05,6,10,23,0,0,0,1,50,0,0,15,0,0.5,0.1,1,0,0,5,2,12,0,0,0,10]
    feat_max=[300,100,50,300,300,300,100,100,40,60,1.5,9,100,100,10000,300,2500,30,150,30,40,1000,35,10,20,15,50,300,70,25,250,250,1200,1500,100]
        
    model=all_model[0]
    model2=all_model[1]
    threshold=all_model[2]
    threshold2=all_model[3]
    length_used=all_model[4]
    min_feat=all_model[5]
    max_feat=all_model[6]
    
    current_features=np.ones([np.size(features,0),40])*-1
    # Remove outliers
    for i in range(0,np.size(features,0)):
                         
        feat_now=features[i,0:35]
        feat_now[feat_now<feat_min]=np.nan
        feat_now[feat_now>feat_max]=np.nan
        features[i,0:35]=feat_now
        feat_now=features[i,:]
        feat_now=(feat_now-min_feat)/(max_feat-min_feat)
        feat_now[np.isnan(feat_now)]=-1
        
        current_features[i,:]=feat_now
        
    for i in range(0,np.size(features,0)):
            test_features=np.ones([1,length_used,40])*-1
            if i<length_used:                
                test_features[0,-(i+1):,:]=current_features[0:i+1,:]
                score=model2.predict(test_features)
                if score>threshold2:
                    label=1
                else:
                    label=0
                    
                score=label
                        
            else:
                test_features[0,:,:]=current_features[i-length_used+1:i+1,:]
                score=model.predict(test_features)
                if score>threshold:
                    label=1
                else:
                    label=0
                
                score=label
            

    return score, label

def load_sepsis_model():
    import numpy as np
    from tensorflow.keras.models import load_model
    
    threshold=np.load('threshold.npy')
    threshold2=np.load('threshold2.npy')
    length_used=np.load('length_used.npy')
    min_feat=np.load('min_feat.npy')
    max_feat=np.load('max_feat.npy')
    model=load_model('model')
    model2=load_model('model2')
    
    all_model = [model,model2,threshold,threshold2,length_used,min_feat,max_feat]
    
    
    return all_model
