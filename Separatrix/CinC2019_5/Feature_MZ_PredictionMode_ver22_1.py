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
import pandas as pd
import numpy as np, os, sys
import scipy.stats
from replace_NaN_MZ_complex import *
from replace_NaN_MZ_interpol import *
import copy
import pickle

def Feature_MZ_PredictionMode_ver22_1(values65, imputer):
    
    
    
    ##*************************************************************************
    FF0_0 = NaN_based_FE_v1(copy.deepcopy(values65)) 
    ##*************************************************************************
    ##*************************************************************************
    values5x, _ = replace_NaN_MZ_complex(copy.deepcopy(values65), 2, imputer)
    ##*************************************************************************
    values5 = replace_NaN_MZ_interpol(copy.deepcopy(values65), copy.deepcopy(values5x)) #11
    
    values5a = values5x[:,[34, 35]]
    values5a = np.concatenate((values5, values5a),axis=1)
    ##*************************************************************************
    FF2 = window_based_FE22(copy.deepcopy(values5), 5) 
    FF5 = window_based_FE22(copy.deepcopy(values5), 11)
    ##*************************************************************************
    ##*************************************************************************
    FF6 = MZ_Features_22_1(copy.deepcopy(values5)) 
    ##*************************************************************************
    ##*************************************************************************
    FF7 = current_based_FE(copy.deepcopy(values5a))
    ##*************************************************************************
    ##*************************************************************************
    Features = np.concatenate((FF0_0, FF2, FF5, FF6, FF7), axis=1)
    del values5, values65, FF0_0, FF2, FF5, FF6, FF7
    ##*************************************************************************
    return Features






##*****************************************************************************
##*****************************************************************************
##*****************************************************************************
def window_based_FE22(values5, Wl):
    
    FF = np.zeros((1, values5.shape[1]*9))
    ##*************************************************************************
    l_z = values5.shape[0]
    if l_z-Wl >= 0:
        F30 = values5
    else:
        l_z_1 = Wl - l_z
        stack11 = np.zeros((l_z_1, values5.shape[1]))
        for iter34 in range(l_z_1):
            stack11[iter34,:] = values5[0,:]
        F30 = np.concatenate((stack11, values5), axis= 0)
        del stack11, l_z_1, l_z
    ##*************************************************************************
    shifted = pd.DataFrame(F30)
    window = shifted.rolling(window=Wl)
    ##*************************************************************************
    A0 = window.mean();   A0 = A0.values;  A0 = A0[-1,:]; A0 = np.expand_dims(A0, axis=0)
    A1 = window.min();    A1 = A1.values;  A1 = A1[-1,:]; A1 = np.expand_dims(A1, axis=0)
    A2 = window.max();    A2 = A2.values;  A2 = A2[-1,:]; A2 = np.expand_dims(A2, axis=0)
    A3 = window.median(); A3 = A3.values;  A3 = A3[-1,:]; A3 = np.expand_dims(A3, axis=0)
    A4 = window.var();    A4 = A4.values;  A4 = A4[-1,:]; A4 = np.expand_dims(A4, axis=0)
    A7 = window.quantile(0.95);    A7 = A7.values;    A7 = A7[-1,:];   A7 = np.expand_dims(A7, axis=0)
    A8 = window.quantile(0.99);    A8 = A8.values;    A8 = A8[-1,:];   A8 = np.expand_dims(A8, axis=0)
    A9 = window.quantile(0.05);    A9 = A9.values;    A9 = A9[-1,:];   A9 = np.expand_dims(A9, axis=0)
    A10 = window.quantile(0.01);   A10 = A10.values;  A10 = A10[-1,:]; A10 = np.expand_dims(A10, axis=0)
    ##*************************************************************************
    ##*************************************************************************
    FF[0, :] = np.concatenate((A0, A1, A2, A3, A4, A7, A8, A9, A10), axis=1)
    ##*************************************************************************
    return FF
    
    
##*****************************************************************************
##*****************************************************************************
##*****************************************************************************
def MZ_Features_22_1(values5):
    
    values1 = values5; del values5
    FF_MZ = np.zeros((1, values1.shape[1]*3))
    A9 = np.sum (np.power(values1, 2), axis=0); A9 = np.expand_dims(A9, axis=0) 
    ##*************************************************************************
    Wl = 3
    l_z = values1.shape[0]
    if l_z-Wl >= 0:
        F30 = values1
    else:
        l_z_1 = Wl - l_z
        stack11 = np.zeros((l_z_1, values1.shape[1]))
        for iter34 in range(l_z_1):
            stack11[iter34,:] = values1[0,:]
        F30 = np.concatenate((stack11, values1), axis= 0)
        del stack11, l_z_1, l_z
    ##*************************************************************************
    ##*************************************************************************
    eps = 2.220446049250313e-16
    pA1 = np.zeros((F30.shape[0], F30.shape[1]))
    for iter12 in range(F30.shape[1]):
        if np.var(F30[:,iter12]) !=0:
            kde = scipy.stats.gaussian_kde(F30[:,iter12])
            pA1[:,iter12] = kde.pdf(F30[:,iter12])
            del kde
        else:
            pA1[:,iter12] = np.ones((1,F30.shape[0]))
    
    pA1 = pA1 + eps
    A10 = -np.sum(pA1*np.log2(pA1), axis = 0); A10 = np.expand_dims(A10, axis=0)
    ##*************************************************************************
    ##*************************************************************************
    df = pd.DataFrame(F30); A11 = df.diff(); A11 = A11.values; A11 = A11[1:,:]
    A11_1 = np.mean(A11, axis= 0); A11_1 = np.expand_dims(A11_1, axis=0) 
    ##*************************************************************************
    ##*************************************************************************
    FF_MZ[0, :] = np.concatenate((A9, A10, A11_1), axis=1)
    ##*************************************************************************
    return FF_MZ

##*****************************************************************************
##*****************************************************************************
##*****************************************************************************
def current_based_FE(values5):

    FF_C = np.zeros((1, values5.shape[1]+1))
    ##*************************************************************************
    F14 = values5.shape[0];    F14 = np.expand_dims(F14,0); F14 = np.expand_dims(F14,0)
    ##*************************************************************************
    Orig_value = values5[-1,:]; Orig_value = np.expand_dims(Orig_value,0) 
    ##*************************************************************************
    FF_C = np.concatenate((F14, Orig_value), axis=1)
    ##*************************************************************************
    return FF_C
    

##*****************************************************************************
##*****************************************************************************
##*****************************************************************************   
def NaN_based_FE_v1(values):
    
    values11 = values
    del values
    ##*************************************************************************
    indx1 = np.append(np.arange(34), np.arange(36,40))
    values_sub = values11[:,indx1]
    ##*************************************************************************
    df = pd.DataFrame(values_sub)
    newdf = df.notnull().astype('int')
    aa = newdf.values
    ##*************************************************************************
    Fx = np.zeros((aa.shape[1], 2))
    num_seq = np.zeros((1, aa.shape[1]))
    ##*************************************************************************
    for iterb in range(aa.shape[1]):
        a = aa[:,iterb]
        a_diff = np.diff(a)
        ind = np.where(a_diff == -1)[0]
        a_diff[ind] = 1
        ind = np.where(a_diff == 1)[0]
        if len(ind) != 0:
            
            if ind[-1] == len(a_diff):
                ind1 = np.concatenate(([0], ind))
                num_seq[0, iterb] = len(ind) 
            else:
                ind1 = np.concatenate(([0], ind, [len(a_diff)]))
                num_seq[0, iterb] = len(ind) + 1
            
            ind1 = np.diff(ind1) 
            ind1[::2]  += 1
            ind1[1::2] -= 1
            
            Fx[iterb, :] = [np.mean(ind1), np.var(ind1)]
            del ind1
        else:
            num_seq[0, iterb] = 1
            Fx[iterb, :] = [0, 0]
            
        del ind, a_diff, a
    Fx1 = np.ndarray.flatten(Fx)
    Fx1 = np.expand_dims(Fx1, axis=0) 
    
    Fx2 = np.sum(aa, axis= 0)/aa.shape[0]  
    Fx2 = np.expand_dims(Fx2, axis=0) 
    
    Fx22 = np.var(aa, axis= 0)/aa.shape[0] 
    Fx22 = np.expand_dims(Fx22, axis=0)
    
    del df, newdf, aa, Fx
    ##*************************************************************************
    ##*************************************************************************
    ##*************************************************************************
    Wl = 5
    l_z = values_sub.shape[0]
    
    if l_z-Wl == 0:
        F30 = values_sub
    elif l_z-Wl > 0:
        F30 = values_sub[-Wl:,:]
    elif l_z-Wl < 0:
        l_z_1 = Wl - l_z
        stack11 = np.zeros((l_z_1, values_sub.shape[1]))
        for iter34 in range(l_z_1):
            stack11[iter34,:] = values_sub[0,:]
        F30 = np.concatenate((stack11, values_sub), axis= 0)
        del stack11, l_z_1, l_z
    df = pd.DataFrame(F30)
    newdf = df.notnull().astype('int')
    aa = newdf.values
    Fx3 = np.sum(aa, axis= 1)/aa.shape[1];  Fx3 = np.expand_dims(Fx3, axis=0) 
    Fx33 = np.var(aa, axis= 1)/aa.shape[1]; Fx33 = np.expand_dims(Fx33, axis=0) 

    FF_D = np.concatenate((Fx1, Fx2, Fx22, Fx3, Fx33), axis=1)
    ##*************************************************************************
    return FF_D