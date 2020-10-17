'''
BSD 2-Clause License

Copyright (c) 2019, PhysioNet
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR
from pykalman import KalmanFilter
import numpy as np

def my_kalman_smooth(window):
	kf = KalmanFilter()
	kf = kf.em(window, n_iter=5)
	(smoothed_state_means, smoothed_state_covariances) = kf.smooth(window)
	return smoothed_state_means[-1]

def my_kalman_pred(window):	
	kf = KalmanFilter()
	means, covariances = kf.filter(window[:-2])
	next_mean, next_covariance = kf.filter_update(
		means[-1], covariances[-1], window[-1])
	return next_mean

def my_kalman_pred_2(window):	
	kf = KalmanFilter()
	means, covariances = kf.filter(window)
	return means[-1]

def my_sarimax(window):
    if np.all(window == np.mean(window)):
        return window[-1]
    model = SARIMAX(window, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1), trend='n', enforce_stationarity=False, le_regression=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False) 
    return model_fit.predict(start=5, end=5)

def my_arma(window):
    if np.unique(window).size == 1:
        return window[-1]
    #print(window)
    mod = ARMA(window, order=(1, 0))
    model_fit = mod.fit(disp=False, transparams=False, full_outputbool=False)
    return model_fit.predict(start=5, end=5)

def my_ar(window):
    if np.unique(window).size == 1:
        return window[-1]
    #print(window)
    mod = AR(window)
    try:
	    model_fit = mod.fit()
    except:
        model_fit = mod.fit(maxlag=2, trend='nc')
    return model_fit.predict(start=5, end=5)
