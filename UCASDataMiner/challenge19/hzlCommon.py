#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on 2018/12/15
@author: ZhenglingHe
""" 
import numpy as np 
from numpy import newaxis
from keras.preprocessing import *
class hzlCommon:
    def classfierIndictor(self,predictLabel,trueLabel):
        TP=0
        FN=0
        FP=0
        TN=0
        for i in range(len(predictLabel)):
            if trueLabel[i]==1 and predictLabel[i]==1:
                TP+=1
            if trueLabel[i]==1 and predictLabel[i]==0:
                FN+=1
            if trueLabel[i]==0 and predictLabel[i]==1:
                FP+=1
            if trueLabel[i]==0 and predictLabel[i]==0:
                TN+=1
        acc=(TP+TN)/(TP+FP+FN+TN)
        sen=TP/ (TP+ FN) #
        spe=TN / (FP + TN)
        FPR=TP/(TP+FN) #
        F1_Score = float(2 * TP) / float(2 * TP + FP + FN)
        print("=======Performance=======")
        print("TP,FN,FP,TN,F1_Score",TP,FN,FP,TN,F1_Score)    
        return [acc,sen,spe]
    def __init__(self):
        self.x_mean = np.array([82.45,94.76,34.12,118.83,80.33,51.02,18.12,2.84,-0.06,10.48,0.22,3.35,17.87,28.17,43.15,18.35,28.78,5.72,48.66,1.14,0.06,112.83,0.58,1.47,2.01,3.34,0.48,0.88,25.50,8.37,16.48,8.73,32.87,159.19,62.00,0.56,0.30,0.31,-54.34,26.91,0.15,-1.85,0.00,0.24,0.77,0.15,-0.19,-0.07,-4.78,-0.01,0.01,0.02,-2.32,-0.01,-0.01,0.01,0.12,0.33,-0.98,0.01,0.04,-0.46,-0.00,0.30,-0.24,0.02,0.13,0.29,0.01,0.39])
        self.x_std = np.array([21.28,15.36,9.70,33.11,21.08,28.74,6.00,9.97,2.29,12.36,6.48,3.68,20.97,43.22,318.48,19.55,70.78,3.95,52.82,1.68,0.75,63.73,1.24,0.97,2.01,1.67,1.98,8.57,13.54,4.67,23.30,7.63,107.75,124.49,16.39,0.50,0.46,0.46,156.56,29.00,0.35,23.52,0.19,0.43,5.90,0.92,22.04,0.59,33.21,0.27,0.63,0.46,17.10,0.21,0.49,0.21,0.16,0.47,15.83,0.19,0.21,5.07,0.12,0.46,4.53,0.31,0.34,0.89,0.02,0.49])
    def uniformDataArray(self,data,axis=0):  
        x_norm = np.nan_to_num((data - self.x_mean) / self.x_std)
        return x_norm 
    def uniformThreeDimen(self,data,type='zscore',axis=0 ):        
        s1=data.shape[0]
        s2=data.shape[1]
        s3=data.shape[2] 
        xx=data.reshape(s1*s2,s3)
        xx=self.uniformDataArray(xx,type,axis)
        return xx.reshape(s1,s2,s3)

    def ratioFeature(self,cur_index,tmp):
        if self.baseLine[cur_index]==-99 and tmp[cur_index]!=0:
            self.baseLine[cur_index]=tmp[cur_index] #Set baseline as current value
            return [0,0]
        elif self.baseLine[cur_index]==-99 and tmp[cur_index]==0:
            return [0,0]
        else:                
            return [tmp[cur_index]-self.baseLine[cur_index],(tmp[cur_index]-self.baseLine[cur_index])*1.0/self.baseLine[cur_index]]       
        #Obatin the ratio. 
    def constructFeaturesAndNormalize(self,values,s,curLen,fillModel="recent"):
        if fillModel not in ["zero","recent"]:
            print("ERROR!!! Unknow data filling model!")
            return 0
        if len(values)==1:
            self.fillData=[]
            self.baseLine=[-99 for i in range(len(values[0]))]
            

        tmp=[]
        recentValue=values[-1]
        for i in range(len(recentValue)):
            if str(recentValue[i])=='nan':
                #print("NULL--")
                if curLen==1:
                    tmp.append(0)
                elif fillModel=="recent":
                    tmp.append(self.fillData[-1][i]) #Use recent value to fill it
                elif fillModel=="zero":
                    tmp.append(0) #Use 0 to fill it
            else:
                tmp.append(recentValue[i])

        sofa=0
        feature_expand=[]
        
        #=======qSOFA=========        
        cur_index=3
        if (0.001<=tmp[cur_index]) and (tmp[cur_index]<=100): # Index 3 is SBP
            feature_expand.append(1)
        else:
            feature_expand.append(0)
        feature_expand+=self.ratioFeature(cur_index,tmp)

        cur_index=6
        if (22<=tmp[cur_index]): # Index 6 is Resp
            feature_expand.append(1)
        else:
            feature_expand.append(0)  
        feature_expand+=self.ratioFeature(cur_index,tmp)
        
        
        
        #=======SOFA=========
        cur_index=10 # Index 10 is Fi02
        feature_expand+=self.ratioFeature(cur_index,tmp)
        
        cur_index=33
        if (100<=tmp[cur_index]) and (tmp[cur_index]<150): # Index 33 is Platelets
            sofa+=1
        elif (50<=tmp[cur_index]) and (tmp[cur_index]<100): 
            sofa+=2
        elif (20<=tmp[cur_index]) and (tmp[cur_index]<50): 
            sofa+=3
        elif (0.001<=tmp[cur_index]) and (tmp[cur_index]<20): 
            sofa+=4
        feature_expand+=self.ratioFeature(cur_index,tmp)

        
        cur_index=26
        if (1.2<=tmp[cur_index]) and (tmp[cur_index]<1.9): # Index 26 is Bilirubin_total
            sofa+=1
        elif (1.9<=tmp[cur_index]) and (tmp[cur_index]<5.9): 
            sofa+=2
        elif (5.9<=tmp[cur_index]) and (tmp[cur_index]<11.9): 
            sofa+=3
        elif (11.9<=tmp[cur_index]): 
            sofa+=4    
        feature_expand+=self.ratioFeature(cur_index,tmp)
        
        cur_index=4
        if (0.001<=tmp[cur_index]) and (tmp[cur_index]<70): # Index 4 is MAP
            sofa+=1
        feature_expand+=self.ratioFeature(cur_index,tmp)    

        cur_index=19
        if (1.2<=tmp[cur_index]) and (tmp[cur_index]<1.9): # Index 19 is Creatinine
            sofa+=1
        elif (1.9<=tmp[cur_index]) and (tmp[cur_index]<3.4): 
            sofa+=2
        elif (3.4<=tmp[cur_index]) and (tmp[cur_index]<4.9): 
            sofa+=3
        elif (4.9<=tmp[cur_index]): 
            sofa+=4       
        feature_expand+=self.ratioFeature(cur_index,tmp)  
        feature_expand.append(sofa/10.0)#sofa, divided by 10

#         print(len(feature_expand)) #result is 17
        
        #=======SIRS========
        cur_index=0
        if (90<tmp[cur_index]): # Index 0 is HR
            feature_expand.append(1)
        else:
            feature_expand.append(0)            
        feature_expand+=self.ratioFeature(cur_index,tmp)
        cur_index=12
        if (0.001<=tmp[cur_index]) and (tmp[cur_index]<32): # Index 12 is PaCo2
            feature_expand.append(1)
        else:
            feature_expand.append(0)            
        feature_expand+=self.ratioFeature(cur_index,tmp)  
        
        cur_index=31
        if ((4<=tmp[cur_index]) and (tmp[cur_index]<12)) or tmp[cur_index]<0.01: # Index 31 is WBC
            feature_expand.append(0)
        else:
            feature_expand.append(1)            
        feature_expand+=self.ratioFeature(cur_index,tmp)  
        
        cur_index=2
        if ((36<=tmp[cur_index]) and (tmp[cur_index]<38)) or tmp[cur_index]<0.01: # Index 2 is Temperature
            feature_expand.append(0)
        else:
            feature_expand.append(1)            
        feature_expand+=self.ratioFeature(cur_index,tmp)  
        
        cur_index=6
        if (20<=tmp[cur_index]): # Index 6 is Resp (The threshold of RESP in SIRS and SOFA is different.)
            feature_expand.append(1)
        else:
            feature_expand.append(0)  
            
        #=======Combine the extended features and raw data========
        tmp+=feature_expand
        self.fillData.append(tmp) #Add data to fillData
#         uX=np.array(self.fillData[s:curLen])  #Only for test
        uX=self.uniformDataArray(np.array(self.fillData[s:curLen]))  #Data normalized 
#         print("====",uX.shape) #result is 70
        return uX

    def constructFeaturesAndNormalizeForWholeFile(self,values,timestamps,fillModel="recent"):
        uX=[]
        for i in range(1,len(values)+1):
            curLen=i
            s=curLen-timestamps
            if s<0:
                s=0
#             print("values[:i]",values[:i])
            uX.append(self.constructFeaturesAndNormalize(values[:i],s,curLen,fillModel))
        return sequence.pad_sequences(np.array(uX), maxlen=timestamps,dtype='float', value=-1,padding='pre')
