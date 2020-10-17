#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Michael Moor July 2019.

RAW-TCN module
"""
import tensorflow as tf
import numpy as np

import sys
sys.path.insert(0, '../../src')

from src.tcn.tcn import CausalConv1D, TemporalBlock, TemporalConvNet
from src.utils.util import compute_global_l2


class RAWTCN():
    """
    RAW-TCN Model: TCN (without MGP imputation) predicts and returns loss
    """
    def __init__(self, tcn_params, input_dim=34, class_imb=1, L2_penalty=None):
        #n_channels, levels, kernel_size, dropout, self.n_classes = tcn_params 
        #unpack tcn parameters    
        n_channels = tcn_params['n_channels']
        levels = tcn_params['levels']
        kernel_size = tcn_params['kernel_size']
        dropout = tcn_params['dropout']
        self.n_classes = tcn_params['n_classes'] 
        self.input_dim = input_dim
        self.tcn = TemporalConvNet([n_channels] * levels, kernel_size, dropout) # initialize TCN     
        self.class_imb = class_imb
        self.L2_penalty = L2_penalty

    def predict(self, minibatch):
        return self._get_preds(minibatch, self.tcn, 
            self.n_classes, self.input_dim )
    
    def probs(self, preds, minibatch):
        probs, mask = self._get_probs(preds, minibatch)
        return probs, mask 
    
    def loss(self, preds, minibatch):
        #TODO: before reduce_sum gather only valid points from cross_entropy tensor (same shape as logits)
        max_len = tf.shape(minibatch.Z)[1]
        mask = tf.sequence_mask(minibatch.valid_points, max_len)
        #get instance and valid timepoint-wise loss 
        instance_and_tp_wise_loss = tf.nn.weighted_cross_entropy_with_logits(
            logits=preds, targets=minibatch.O_dupe_onehot,  pos_weight=self.class_imb)
        instance_and_tp_wise_loss = tf.squeeze(instance_and_tp_wise_loss, -1) #remove last dimension which is [1]
        masked_loss = tf.cast(mask, tf.float32) * instance_and_tp_wise_loss
        instance_wise_loss = tf.reduce_sum(masked_loss, axis=-1) \
            / tf.cast(minibatch.valid_points, tf.float32)
        loss = tf.reduce_mean(instance_wise_loss)
        if self.L2_penalty is not None:
            loss_reg = compute_global_l2() # normalized per weight! hence, use large lambda!
            loss = loss + loss_reg * self.L2_penalty
        return loss
   
    def load(self, sess, restore_path):
        """
        Function to restore model state as saved in path
        """    
        saver = tf.train.Saver(max_to_keep = None)        
        saver.restore(sess, restore_path) 
        return None

    def _get_preds(self, minibatch, tcn, n_classes, input_dim): 
        """
        helper function. feeds raw padded data through TCN to get predictions.

        inputs:
            - minibatch object, containing Z: padded minibatch array of time series (labs/vitals). batchsize x batch_maxlen x M
            - TCN object
            - is_training flag (whether tcn should apply dropout or not)

        returns:
            predictions (unnormalized log probabilities) for each minibatch sample
        """
        Z = minibatch.Z 

        Z.set_shape([None,None,input_dim]) #somehow lost shape info, but need this
        tcn_output = tcn(Z, training=minibatch.is_training) #[:,-1,:] for last output, [batchsize, time, n_channels] 

        # TODO: N-classes should be 1 if we are doing timepointwise prediction of a binary label
        tcn_logits = tf.layers.dense(
            tcn_output,
            n_classes, activation=None,
            kernel_initializer=tf.zeros_initializer(),
            name='last_linear')
        #TODO: determine new ouput shape of logits and adjust 
        return tcn_logits         

    def _get_probs(self, preds, minibatch):
        """
        helper function. get predicted prob from logits.
        Inputs: - logits (unmasked, still padded with invalid points)
        Returns - probability scores (padded) 
                - boolean mask for valid probability scores
        """
        max_len = tf.shape(minibatch.Z)[1] #maximal number of time points of current minibatch
        mask = tf.sequence_mask(minibatch.valid_points, max_len) # boolean mask, 
        # indicating valid time points of the padded minibatch tensor
 
        #all_probs = tf.exp(preds[:,:,0] - tf.reduce_logsumexp(preds, axis = 2)) #normalize; and drop a dim so only prob of positive case
        all_probs = tf.nn.sigmoid(preds)         

        return all_probs, mask
  
class Minibatch():
    """ Minibatch Object containing tensorflow placeholders for data for feeding in the graph
    """
    def __init__(self, n_classes):
        #n_mc_smps = 1 # set this as dummy var to use same O_dupe_onhot as in MGP
        self.Z = tf.placeholder("float", [None,None,None]) #TCN inputs 
        self.O = tf.placeholder(tf.int32, [None, None]) #labels. input is NOT as one-hot encoding; convert at each iter
        self.N = tf.shape(self.Z)[0]  # number of samples in minibatch                       
        #TODO: adjust O_dupe_onehot for additional dim
        #self.O_dupe_onehot = tf.one_hot(tf.reshape(tf.tile(tf.expand_dims(self.O,1),[1,n_mc_smps]),[self.N*n_mc_smps]), n_classes)
        # self.O_dupe_onehot = tf.one_hot(self.O, n_classes)
        self.O_dupe_onehot = tf.cast(tf.expand_dims(self.O, -1), tf.float32) #pseudo one_hot encoding as crossentropy requires this format

        self.is_training = tf.placeholder("bool")
        self.valid_points = tf.placeholder(tf.int32, [None]) # indicates the numb of valid points of the padded time series per patient
                                                                                                                     

             
                                                    






##old implementation to average the output sequence:
#def get_losses(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
#              num_tcn_grid_times, cov_grid, input_dim,method, gp_params, tcn, is_training, n_classes, pad_before,
#              labels, pos_weight): #med_cov_grid
#    """
#    helper function. takes in (padded) raw datas, samples MGP for each observation, 
#    then feeds it all through the TCN to get predictions.
#    
#    inputs:
#        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
#        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
#        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
#        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
#        num_obs_times: number of times observed for each encounter; how long each row of T really is
#        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
#        num_tcn_grid_times: length of even spaced TCN grid per encounter
#                    
#    returns:
#        predictions (unnormalized log probabilities) for each MC sample of each obs
#    """
#    
#    Z = get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
#                       num_tcn_grid_times, cov_grid, input_dim, method=method, gp_params=gp_params, pad_before=pad_before) #batchsize*num_MC x batch_maxseqlen x num_inputs    ##,med_cov_grid
#    Z.set_shape([None,None,input_dim]) #somehow lost shape info, but need this
#    N = tf.shape(Z)[0] #number of observations 
#
#
#    # We only want to consider up tw0 7 timepoints before end
#    T_max = 7
#    
#    # Only during training we want to average over the last few predictions in order to give
#    # the model the incentive to predict early
#    tcn_out = tcn(Z, training=is_training)[:, -T_max:, :]
#    tcn_logits = tf.layers.dense(tcn_out,
#        n_classes, activation=None, 
#        kernel_initializer=tf.orthogonal_initializer(),
#        name='last_linear', reuse=False
#    )
#
#    # Only get a few of the last obs
#    #used_grid = tf.reduce_min(tf.stack([num_tcn_grid_times, tf.fill(tf.shape(num_tcn_grid_times), T_max)]), axis=0)
#    #tiled = tf.tile(tf.expand_dims(used_grid, axis=-1), [1, gp_params.n_mc_smps])
#    #expanded_used_grid = tf.reshape(tiled, [-1])
#    tiled_labels = tf.tile(tf.expand_dims(labels, axis=1), tf.stack([1, T_max, 1]))
#    all_losses = tf.nn.weighted_cross_entropy_with_logits(logits=tcn_logits,targets=tiled_labels, pos_weight=pos_weight)
#    average_losses = tf.reduce_mean(all_losses, axis=-1)
#
#    return average_losses

