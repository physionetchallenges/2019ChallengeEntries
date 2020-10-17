#!/usr/bin/env python

import numpy as np
from src.raw_tcn.util import standardize, impute, read_configs
#from src.raw_tcn.trained_model import Fitted_Model
from src.lstm.trained_model import Fitted_Model

def get_sepsis_score(data, model):

    #Standardize data (convert to pd df)
    x = data[:,:40] #34 up to model 31
    z = standardize(x) #z is now a pd df!
   
    #Impute missing values
    z = impute(z)
    
    #Convert to np array
    z_array = z.values  
    
    #Feed data into model
    score, label = model(z_array)

    #4. Return Score, Predicted Label
    return score, label

def load_sepsis_model():
    #load sacred config json here, and overwrite bs=1
    path = 'runs/lstm/4' #53' #lstm/9' #8' #7' #lstm/6' # lstm/3 -- tcn: runs/40' #35' #33' #31 #23, #20, 22
    config_path = path + '/config.json'
    configs = read_configs(config_path)
    tcn_parameters = configs['tcn_parameters'] 
    
    #set restore path of desired model checkpoint
    #   we try model checkpoint run 20: '/epoch_18-3000'
    restore_path = path + '/model_checkpoints/epoch_12-980' #TCN 53:  epoch_3-320' #lstm run9: epoch_1-120'  #run8: epoch_2-240'  #run7: epoch_0-60' #lstm run6: epoch_1-140' # lstm run3: epoch_8-1400' #TCNs: run 40: epoch_4-760' #run 35: epoch_14-2400' #run 33: epoch_11-1920' #run31: epoch_13-2240' #run23: epoch_41-6720' #run22: epoch_6-1000'  
    #initialize fitted model class:
    model = Fitted_Model(tcn_parameters, restore_path) 
    print('Model loaded!')
 
    return model
