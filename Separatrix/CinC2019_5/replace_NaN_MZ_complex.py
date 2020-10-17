#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Morteza Zabihi (morteza.zabihi@gmail.com) 
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
import pandas as pd
from sklearn.impute import SimpleImputer
 
 
def replace_NaN_MZ_complex(values, Phase, imputer):
    
    
    indx1 = np.append(np.arange(34), np.arange(36,40))
    indx2 = 34
    indx3 = 35
    ###########################################################################
    if Phase == 1: # Training Phase
        imputer =  {}
        feature_1 = values[:,indx1]
        feature_2 = values[:,indx2]; feature_2 = feature_2.reshape(-1, 1)
        feature_3 = values[:,indx3]; feature_3 = feature_3.reshape(-1, 1)
        #######################################################################        
        imputer1 = SimpleImputer(missing_values=np.nan,strategy="mean")
        imputer2 = SimpleImputer(missing_values=np.nan,strategy="mean")
        imputer3 = SimpleImputer(missing_values=np.nan,strategy="mean")
        #######################################################################
        imputer1.fit(feature_1)
        imputer2.fit(feature_2)
        imputer3.fit(feature_3)
        #######################################################################
        feature_1 = imputer1.transform(feature_1)
        feature_2 = imputer2.transform(feature_2)
        feature_3 = imputer3.transform(feature_3)
        #######################################################################
        values[:,indx1] = feature_1
        values[:,indx2] = np.squeeze(feature_2)
        values[:,indx3] = np.squeeze(feature_3)
        #######################################################################
        imputer['inner_Imputer1'] = imputer1
        imputer['inner_Imputer2'] = imputer2
        imputer['inner_Imputer3'] = imputer3
        #######################################################################
        #######################################################################
        #######################################################################
    elif Phase == 2: # Testing Phase
        
        imputer1 = imputer['inner_Imputer1']
        imputer2 = imputer['inner_Imputer2']
        imputer3 = imputer['inner_Imputer3']
        #######################################################################
        feature_1 = values[:,indx1]
        feature_2 = values[:,indx2]; feature_2 = feature_2.reshape(-1, 1)
        feature_3 = values[:,indx3]; feature_3 = feature_3.reshape(-1, 1)
        #######################################################################          
        feature_1 = imputer1.transform(feature_1) 
        #######################################################################
        L = feature_2.shape[0]
        stack1 = np.where(np.isnan(feature_2) == True)[0]
        #Condition 1: In case all the availabe values are NAN:
        if len(stack1) == L:
            feature_2 = imputer2.transform(feature_2)
        else:
            df = pd.DataFrame(feature_2)
            val1 = df.first_valid_index()
            val2 = feature_2[val1]
            ones1 = np.ones((feature_2.shape[0], feature_2.shape[1]))
            feature_2 = ones1*val2
            del df, val1, val2, ones1
        del stack1
        #######################################################################
        L = feature_3.shape[0]
        stack1 = np.where(np.isnan(feature_3) == True)[0]
        #Condition 1: In case all the availabe values are NAN:
        if len(stack1) == L:
            feature_3 = imputer3.transform(feature_3)
        else:
            df = pd.DataFrame(feature_3)
            val1 = df.first_valid_index()
            val2 = feature_3[val1]
            ones1 = np.ones((feature_3.shape[0], feature_3.shape[1]))
            feature_3 = ones1*val2    
            del df, val1, val2, ones1
        del stack1
        #######################################################################
        values[:,indx1] = feature_1
        values[:,indx2] = np.squeeze(feature_2)
        values[:,indx3] = np.squeeze(feature_3)
        #######################################################################
        #######################################################################
        #######################################################################
    #######################################################################
    return values, imputer
            
            
            
            
            
            
            
        
    