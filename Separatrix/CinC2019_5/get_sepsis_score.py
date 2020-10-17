#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:
Morteza Zabihi (morteza.zabihi@gmail.com) 
(June 2019)
=============================================================================== 
This code is released under the GNU General Public License.
You should have received a comodelspy of the GNU General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

By accessing the code through Physionet webpage and/or by installing, copying, or otherwise
using this software, you agree to be bounded by the terms of GNU General Public License.
If you do not agree, do not install copy or use the software.

We make no representation about the suitability of this licensed deliverable for any purppose.
It is provided "as is" without express or implied warranty of any kind.
Any use of the licensed deliverables must include the above disclaimer.
For more detailed information please refer to "readme" file.
===============================================================================
"""
import numpy as np, os, sys
from sklearn.preprocessing import StandardScaler
import pickle
import scipy 

import xgboost
from Feature_MZ_PredictionMode_ver22_1 import *
import copy


def get_sepsis_score(values, model):
    
    X_train_0 = np.zeros((1, 1))
    ###########################################################################
    FS = model['FS']
    ###########################################################################
    name_imputer = '_Final_imputer_fold_' + str(0)
    imputer = model[name_imputer]
    
    Features_0 = Feature_MZ_PredictionMode_ver22_1(values, imputer['imputer'])
    ###########################################################################
    ###########################################################################
    X_train_0[0,0] = helper_1_1(model, copy.deepcopy(Features_0), FS) 
    ###########################################################################    
    ###########################################################################  
    scores = X_train_0[0,0]   
    labels = (scores > 0.5).astype(np.int_)
    ###########################################################################
    ###########################################################################
    del Features_0
    return scores, labels

###############################################################################
def helper_1_1(model, Features, FS):
    
    iter001 = 0
    ##---------------------------------------------------------------------
    name_scaler = '_Final_scaler_fold_' + str(iter001)
    SCALER_10F = model[name_scaler]
    scaler0 = SCALER_10F['scaler']
    del name_scaler
    Features1 = scaler0.transform(copy.deepcopy(Features))
    ##---------------------------------------------------------------------
    name_model = '_Final_MODEL_inner_'+ str(iter001)
    MODEL_10F = model[name_model]
    del name_model
    #######################################################################
    X_train_1 = np.zeros((1, len(MODEL_10F)))
    #######################################################################
    #######################################################################
    for iter01 in range(len(MODEL_10F)):
        name_model = 'model' + str(iter01)           
        ub0 = MODEL_10F[name_model]
        ##-----------------------------------------------------------------
        indxyz1 = FS['FS'+str(iter01)]
        ##-----------------------------------------------------------------
        probas1_1 = ub0.predict_proba(Features1[:,indxyz1]);
        probas1_1 = np.expand_dims(probas1_1[:,1], 1)
        ##-----------------------------------------------------------------
        X_train_1[:,iter01] = probas1_1
        ##-----------------------------------------------------------------
        del name_model, ub0, probas1_1, indxyz1
        ####################################################################### 
    del SCALER_10F, MODEL_10F, scaler0, Features
    return scipy.stats.mstats.gmean(X_train_1, axis=1)
    

###############################################################################
def load_sepsis_model():
    
    model = {}
    ###########################################################################
    for iter123 in range(1):
        
        name_imputer = '_Final_imputer_fold_' + str(iter123)
        imputer_temp = pickle.load(open(name_imputer, "rb"))
        model[name_imputer] = imputer_temp
        del name_imputer, imputer_temp
        
        name_model = '_Final_MODEL_inner_'+ str(iter123)
        model_temp = pickle.load(open(name_model, "rb"))
        model[name_model] = model_temp
        del name_model, model_temp
        
        name_scaler = '_Final_scaler_fold_' + str(iter123)
        scaler_temp = pickle.load(open(name_scaler, "rb"))
        model[name_scaler] = scaler_temp
        del name_scaler, scaler_temp
        
        name_fs = 'FS'
        fs_temp = pickle.load(open(name_fs, "rb"))
        model[name_fs] = fs_temp
        del name_fs, fs_temp
        
    ###########################################################################
    return model



