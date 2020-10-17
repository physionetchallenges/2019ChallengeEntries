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

def replace_NaN_MZ_interpol(values65, values5x):
    
    values_simple = values5x[:,[0, 1, 2, 3, 4, 5, 6, 36, 37, 38, 39]] 
    values_orig = values65[:,[0, 1, 2, 3, 4, 5, 6, 36, 37, 38, 39]] 
    ##*************************************************************************
    L = values_simple.shape[0]
    ##*************************************************************************
    if L <3:
        Final_val = values_simple
    ##*************************************************************************
    else:
        Final_val = np.zeros((values_orig.shape[0], values_orig.shape[1]))
        
        for iterinter in range(values_orig.shape[1]):
            temp = values_orig[:,iterinter]
            values_orig_pd = pd.DataFrame(temp)
            df = values_orig_pd.assign(InterpolateLinear=values_orig_pd.interpolate(method='linear')) #limit_direction= 'both'
            df = df.values
            Final_val[:,iterinter] = df[:,1]
            del df, values_orig_pd, temp
            
        indx = np.argwhere(np.isnan(Final_val))
        Final_val[indx[:,0], indx[:,1]] = values_simple[indx[:,0], indx[:,1]]
        del indx
    ##*************************************************************************
    return Final_val

        
    