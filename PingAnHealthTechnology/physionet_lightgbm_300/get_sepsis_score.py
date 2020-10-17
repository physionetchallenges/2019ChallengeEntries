#!/usr/bin/env python

import numpy as np
from sklearn.externals import joblib
from lightgbm import LGBMClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def create_feature(current_record):
    col_list=[0,1,2,3,4,5,6,10,13,15,16,17,19,21,22,24,28,29,30,31,33]    
    col_list2=[0,1,2,3,4,5,6,21,31] 

    current_record[:,2]=np.where(current_record[:,2]>42,42,current_record[:,2])
    current_record[:,2]=np.where(current_record[:,2]<32,32,current_record[:,2])
    
    current_record[:,3]=np.where(current_record[:,3]>200,200,current_record[:,3])
    current_record[:,3]=np.where(current_record[:,3]<80,80,current_record[:,3]) 
    
    current_record[:,5]=np.where(current_record[:,5]>130,130,current_record[:,5])
    current_record[:,5]=np.where(current_record[:,5]<50,50,current_record[:,5])   
    
    current_record[:,10]=np.where(current_record[:,10]<0.2,0.2,current_record[:,10])    
            
    curr_value=current_record[-1,:40].reshape(1,-1)   
    SaO2_FiO2 =(current_record[-1,13]/current_record[-1,10]).reshape(1,-1)
    HR_SBP = (current_record[-1,0]/current_record[-1,3]).reshape(1,-1)
    
    temp_age=(current_record[-1,2]/current_record[-1,34]).reshape(1,-1)
    sbp_age=(current_record[-1,3]/current_record[-1,34]).reshape(1,-1)        
    Glucose_age=(current_record[-1,21]/current_record[-1,34]).reshape(1,-1)  
    
    AdmTime_ICU = (current_record[-1,38]-current_record[-1,39]).reshape(1,-1) 
    ICU_AdmTime = (current_record[-1,39]-current_record[-1,38]).reshape(1,-1) 
    
    if len(current_record)==1:          
        max_pre_6h=current_record[-1,col_list].reshape(1,-1)
        min_pre_6h=current_record[-1,col_list].reshape(1,-1)
        
        max_pre_12h=current_record[-1,col_list].reshape(1,-1)
        min_pre_12h=current_record[-1,col_list].reshape(1,-1)     
    
        max_pre_24h=current_record[-1,col_list].reshape(1,-1)
        min_pre_24h=current_record[-1,col_list].reshape(1,-1) 
                
        mean_pre_6h=current_record[-1,col_list2].reshape(1,-1)
        std_pre_6h=np.zeros((1,len(col_list2)))

        mean_pre_12h=current_record[-1,col_list2].reshape(1,-1)
        std_pre_12h=np.zeros((1,len(col_list2)))       

        mean_pre_24h=current_record[-1,col_list2].reshape(1,-1)
        std_pre_24h=np.zeros((1,len(col_list2)))
                         
    elif len(current_record)>1 and len(current_record)<6:       
        max_pre_6h=np.nanmax(current_record[:,col_list],axis=0).reshape(1,-1)
        min_pre_6h=np.nanmin(current_record[:,col_list],axis=0).reshape(1,-1)
        
        max_pre_12h=np.nanmax(current_record[:,col_list],axis=0).reshape(1,-1)
        min_pre_12h=np.nanmin(current_record[:,col_list],axis=0).reshape(1,-1)     
    
        max_pre_24h=np.nanmax(current_record[:,col_list],axis=0).reshape(1,-1)
        min_pre_24h=np.nanmin(current_record[:,col_list],axis=0).reshape(1,-1)
               
        mean_pre_6h=np.nanmean(current_record[:,col_list2],axis=0).reshape(1,-1)
        std_pre_6h=np.nanstd(current_record[:,col_list2],axis=0).reshape(1,-1)

        mean_pre_12h=np.nanmean(current_record[:,col_list2],axis=0).reshape(1,-1)
        std_pre_12h=np.nanstd(current_record[:,col_list2],axis=0).reshape(1,-1)       

        mean_pre_24h=np.nanmean(current_record[:,col_list2],axis=0).reshape(1,-1)
        std_pre_24h=np.nanstd(current_record[:,col_list2],axis=0).reshape(1,-1)  
           
     
    elif len(current_record)>=6 and len(current_record)<12:
        max_pre_6h=np.nanmax(current_record[-6:,col_list],axis=0).reshape(1,-1)
        min_pre_6h=np.nanmin(current_record[-6:,col_list],axis=0).reshape(1,-1)
        
        max_pre_12h=np.nanmax(current_record[:,col_list],axis=0).reshape(1,-1)
        min_pre_12h=np.nanmin(current_record[:,col_list],axis=0).reshape(1,-1)     
    
        max_pre_24h=np.nanmax(current_record[:,col_list],axis=0).reshape(1,-1)
        min_pre_24h=np.nanmin(current_record[:,col_list],axis=0).reshape(1,-1)
        
        
        mean_pre_6h=np.nanmean(current_record[-6:,col_list2],axis=0).reshape(1,-1)
        std_pre_6h=np.nanstd(current_record[-6:,col_list2],axis=0).reshape(1,-1)

        mean_pre_12h=np.nanmean(current_record[:,col_list2],axis=0).reshape(1,-1)
        std_pre_12h=np.nanstd(current_record[:,col_list2],axis=0).reshape(1,-1)       

        mean_pre_24h=np.nanmean(current_record[:,col_list2],axis=0).reshape(1,-1)
        std_pre_24h=np.nanstd(current_record[:,col_list2],axis=0).reshape(1,-1)  
                    
    elif  len(current_record)>=12 and len(current_record)<24:      
        max_pre_6h=np.nanmax(current_record[-6:,col_list],axis=0).reshape(1,-1)
        min_pre_6h=np.nanmin(current_record[-6:,col_list],axis=0).reshape(1,-1)
        
        max_pre_12h=np.nanmax(current_record[-12:,col_list],axis=0).reshape(1,-1)
        min_pre_12h=np.nanmin(current_record[-12:,col_list],axis=0).reshape(1,-1)     
    
        max_pre_24h=np.nanmax(current_record[:,col_list],axis=0).reshape(1,-1)
        min_pre_24h=np.nanmin(current_record[:,col_list],axis=0).reshape(1,-1)
                
        mean_pre_6h=np.nanmean(current_record[-6:,col_list2],axis=0).reshape(1,-1)
        std_pre_6h=np.nanstd(current_record[-6:,col_list2],axis=0).reshape(1,-1)

        mean_pre_12h=np.nanmean(current_record[-12:,col_list2],axis=0).reshape(1,-1)
        std_pre_12h=np.nanstd(current_record[-12:,col_list2],axis=0).reshape(1,-1)       

        mean_pre_24h=np.nanmean(current_record[:,col_list2],axis=0).reshape(1,-1)
        std_pre_24h=np.nanstd(current_record[:,col_list2],axis=0).reshape(1,-1)  
           
        
    elif len(current_record)>=24:
        
        max_pre_6h=np.nanmax(current_record[-6:,col_list],axis=0).reshape(1,-1)
        min_pre_6h=np.nanmin(current_record[-6:,col_list],axis=0).reshape(1,-1)
        
        max_pre_12h=np.nanmax(current_record[-12:,col_list],axis=0).reshape(1,-1)
        min_pre_12h=np.nanmin(current_record[-12:,col_list],axis=0).reshape(1,-1)     
    
        max_pre_24h=np.nanmax(current_record[-24:,col_list],axis=0).reshape(1,-1)
        min_pre_24h=np.nanmin(current_record[-24:,col_list],axis=0).reshape(1,-1)
        
        
        mean_pre_6h=np.nanmean(current_record[-6:,col_list2],axis=0).reshape(1,-1)
        std_pre_6h=np.nanstd(current_record[-6:,col_list2],axis=0).reshape(1,-1)

        mean_pre_12h=np.nanmean(current_record[-12:,col_list2],axis=0).reshape(1,-1)
        std_pre_12h=np.nanstd(current_record[-12:,col_list2],axis=0).reshape(1,-1)       

        mean_pre_24h=np.nanmean(current_record[-24:,col_list2],axis=0).reshape(1,-1)
        std_pre_24h=np.nanstd(current_record[-24:,col_list2],axis=0).reshape(1,-1)  
      
    diff_pre_6h = max_pre_6h- min_pre_6h
    diff_pre_12h =max_pre_12h-min_pre_12h
    diff_pre_24h = max_pre_24h- min_pre_24h

    curr_vitalsign=current_record[-1,col_list2].reshape(1,-1)
    
    if len(current_record)==1:
        pre_1h=current_record[-1,col_list2].reshape(1,-1) 
        pre_2h=current_record[-1,col_list2].reshape(1,-1)
    elif len(current_record)==2:
        pre_1h=current_record[-2,col_list2].reshape(1,-1) 
        pre_2h=current_record[-2,col_list2].reshape(1,-1)
    elif len(current_record)>2:
        pre_1h=current_record[-2,col_list2].reshape(1,-1) 
        pre_2h=current_record[-3,col_list2].reshape(1,-1)
               
    diff_1h= curr_vitalsign-pre_1h 
    diff_2h= curr_vitalsign-pre_2h

    slope_1h=diff_1h/pre_1h
    slope_2h=diff_2h/pre_2h
        
    new_feature=np.hstack((curr_value,SaO2_FiO2,HR_SBP,temp_age,sbp_age,Glucose_age,AdmTime_ICU,ICU_AdmTime,
                           max_pre_6h,min_pre_6h,max_pre_12h,min_pre_12h,max_pre_24h,min_pre_24h,
                           diff_pre_6h,diff_pre_12h, diff_pre_24h,
                           mean_pre_6h,std_pre_6h, mean_pre_12h,std_pre_12h, mean_pre_24h,std_pre_24h,
                           pre_1h, pre_2h,diff_1h,diff_2h,slope_1h,slope_2h))  
    return  new_feature



def get_sepsis_score(data, model):
    col_select=[38, 39, 34, 14, 42, 26, 31, 9, 46, 134, 151, 45, 133, 156, 144, 158, 18, 154, 36, 274, 164, 49, 23, 165, 147,
            171, 2, 12, 7, 136, 163, 146, 148, 152, 273, 142, 167, 170, 11, 217, 44, 231, 172, 25, 277, 155, 161, 235, 168,
            278, 143, 218, 150, 149, 8, 128, 109, 131, 345, 114, 102, 275, 228, 272, 279, 91, 95, 145, 112, 220, 129, 234, 
            169, 123, 119, 226, 41, 104, 227, 116, 89, 287, 255, 135, 140, 27, 19, 137, 215, 230, 141, 121, 40, 281, 166, 162,
            125, 88, 92, 221, 30, 32, 94, 232, 276, 105, 286, 93, 160, 81, 261, 233, 307, 260, 33, 113, 157, 181, 153, 107, 35,
            72, 101, 110, 67, 100, 71, 59, 58, 280, 344, 219, 96, 79, 76, 108, 196, 103, 122, 205, 127, 201, 15, 106, 187, 66, 
            73, 86, 224, 63, 259, 65, 242, 130, 60, 53, 84, 282, 124, 22, 283, 17, 222, 138, 288, 111, 24, 28, 159, 126, 47, 269,
            213, 0, 197, 180, 285, 77, 62, 257, 120, 10, 80, 132, 83, 254, 256, 54, 82, 199, 85, 248, 50, 229, 16, 139, 87, 37, 
            216, 284, 56, 118, 97, 210, 265, 61, 51, 98, 70, 43, 223, 99, 209, 29, 200, 69, 64, 57, 74, 176, 68, 250, 21, 117, 
            306, 289, 55, 194, 208, 207, 264, 262, 243, 20, 115, 212, 52, 237, 13, 298, 175, 78, 198, 177, 263, 268, 266, 202, 
            258, 246, 211, 174, 225, 185, 267, 241, 203, 249, 239, 206, 75, 240, 271, 184, 292, 236, 244, 214, 195, 178, 251, 
            301, 297, 247, 4, 238, 270, 189, 6, 343, 48, 173, 325, 5, 317, 253, 204, 90, 245, 290, 190, 186, 192, 193, 1, 179, 
            299, 296, 294, 337]

    N=len(data)
    current_patient=pd.DataFrame(data)
    masking=1-np.isnan(current_patient).astype(int)

    current_patient=current_patient.fillna(method='ffill')

    current_record=np.array(current_patient)

    if N<24:
        count_sum=np.sum(np.array(masking.iloc[:,:34])).reshape(1,-1)
        count_mean=(np.sum(np.array(masking.iloc[:,:34]))/N).reshape(1,-1)
    else:  
        count_sum=np.sum(np.array(masking.iloc[N-24:N,:34])).reshape(1,-1)       
        count_mean=(np.sum(np.array(masking.iloc[N-24:N,:34]))/24).reshape(1,-1)

    structure_feature=create_feature(current_record) 
    feature_total=np.hstack((structure_feature,count_sum,count_mean))
    feature_total=feature_total[:,col_select]

    scores=model.predict(feature_total)
    labels=(scores>0.023)
    return (scores, labels)

def load_sepsis_model():
    xgb_model=joblib.load('./lightgbm_newfeature_300.pkl')
    return xgb_model
