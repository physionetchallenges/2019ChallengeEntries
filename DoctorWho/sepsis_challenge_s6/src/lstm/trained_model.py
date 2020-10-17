#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Michael Moor October 2018.

"""
#TODO: ADJUST FOR LSTM (hacky)


import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../../src')
sys.path.insert(0, '../../scripts')

#Loading local modules
from src.utils.util import get_pointwise_preds_and_probs 
from src.raw_tcn.util import pad_data_online as pad_data
from src.models.lstm import LSTM_Model, Minibatch


#model checkpoint run 20: '.../epoch_18-3000'

class Fitted_Model():
    """
    Fitted LSTM Model for online predictions
    """
    def __init__(self, tcn_parameters, restore_path, min_pad_length=8, n_classes=1, input_dim=74):
        self.sess = tf.Session()
        self.mb = Minibatch(n_classes)
        self.input_dim = input_dim
        self.tcn_parameters = tcn_parameters
        self.restore_path = restore_path 
        
        #compute min_pad_len
        self.min_pad_length = min_pad_length
    
        #init and load model:
        self.model = LSTM_Model(self.tcn_parameters, self.input_dim)   
        
        #define tf prediction graph:
        self.preds = self.model.predict(self.mb) #model takes minibatch object and returns predictions
        self.probs, self.mask = self.model.probs(self.preds, self.mb) #get probabilities (padded) and mask indicating valid indices)

        self.model.load(self.sess, self.restore_path)


    def __call__(self, data):
        data_pad, valid_points = pad_data(data, self.min_pad_length) 
        
        feed_dict={ self.mb.Z: data_pad, 
                    self.mb.valid_points: valid_points,
                    self.mb.is_training: False
                  }
        pat_prob, pat_mask = self.sess.run([self.probs, self.mask], feed_dict)
        # pointwise probs takes and returns lists!
        preds, probs = get_pointwise_preds_and_probs([pat_prob ], [pat_mask] )
        return probs[0][-1,0], preds[0][-1,0]
        

    

