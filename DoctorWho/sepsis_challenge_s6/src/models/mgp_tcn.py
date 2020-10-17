#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Michael Moor April 2019.

MGP-TCN module
"""
import sys
sys.path.insert(0, '../../src')
import tensorflow as tf
import numpy as np
from mgp.mgp import get_GP_samples 
from tcn.tcn import CausalConv1D, TemporalBlock, TemporalConvNet
from utils.util import compute_global_l2


class MGPTCN():
    """
    MGP-TCN Model: takes TCN, MGP params and data and predicts and returns loss
    """
    def __init__(self, tcn_params, gp_params, class_imb, L2_penalty):
        #n_channels, levels, kernel_size, dropout, self.n_classes = tcn_params 
        #unpack tcn parameters    
        n_channels = tcn_params['n_channels']
        levels = tcn_params['levels']
        kernel_size = tcn_params['kernel_size']
        dropout = tcn_params['dropout']
        self.n_classes = tcn_params['n_classes'] 
        self.tcn = TemporalConvNet([n_channels] * levels, kernel_size, dropout) # initialize TCN     
        self.gp_params = gp_params
        self.class_imb = class_imb
        self.L2_penalty = L2_penalty

    def predict(self, minibatch):
        return get_preds(minibatch, self.gp_params, self.tcn, 
            self.n_classes )
    
    def probs_and_accuracy(self, preds, minibatch ):
        probs, accuracy = get_probs_and_accuracy(preds, minibatch.O, self.gp_params.n_mc_smps)
        return probs, accuracy 
    
    def loss(self, preds, minibatch):
        loss = tf.reduce_sum(
                tf.nn.weighted_cross_entropy_with_logits(
                    logits=preds, targets=minibatch.O_dupe_onehot, pos_weight=self.class_imb)
                    ) 
        if self.L2_penalty is not None:
            loss_reg = compute_global_l2() # normalized per weight! hence, use large lambda!
            loss = loss + loss_reg * self.L2_penalty
        return loss
    

class Minibatch():
    """ Minibatch Object containing tensorflow placeholders for data for feeding in the graph
    """
    def __init__(self, n_mc_smps, n_classes):
        
        #observed values, times, inducing times; padded to longest in the batch
        self.Y = tf.placeholder("float", [None,None]) #batchsize x batch_maxdata_length
        self.T = tf.placeholder("float", [None,None]) #batchsize x batch_maxdata_length
        self.ind_kf = tf.placeholder(tf.int32, [None,None]) #index tasks in Y vector
        self.ind_kt = tf.placeholder(tf.int32, [None,None]) #index inputs in Y vector
        self.X = tf.placeholder("float", [None,None]) #grid points. batchsize x batch_maxgridlen
        #self.cov_grid = tf.placeholder("float", [None,None,n_covs]) #combine w GP smps to feed into TCN ## n_meds+n_covs
        
        self.O = tf.placeholder(tf.int32, [None]) #labels. input is NOT as one-hot encoding; convert at each iter
        self.num_obs_times = tf.placeholder(tf.int32, [None]) #number of observation times per encounter 
        self.num_obs_values = tf.placeholder(tf.int32, [None]) #number of observation values per encounter 
        self.num_tcn_grid_times = tf.placeholder(tf.int32, [None]) #length of each grid to be fed into TCN in minibatch
        
        self.N = tf.shape(self.Y)[0]  # number of samples in minibatch                       
             
        #also make O one-hot encoding, for the loss function
        self.O_dupe_onehot = tf.one_hot(tf.reshape(tf.tile(tf.expand_dims(self.O,1),[1,n_mc_smps]),[self.N*n_mc_smps]), n_classes)

        self.is_training = tf.placeholder("bool")
                                                                                                                                                                                         
def get_preds(minibatch, gp_params, tcn,  n_classes ): #med_cov_grid
    """
    helper function. takes in (padded) raw datas, samples MGP for each observation, 
    then feeds it all through the TCN to get predictions.
    
    inputs:
        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
        num_obs_times: number of times observed for each encounter; how long each row of T really is
        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
        num_tcn_grid_times: length of even spaced TCN grid per encounter
                    
    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """
    
    Z = get_GP_samples(minibatch, gp_params) #batchsize*num_MC x batch_maxseqlen x num_inputs    ##,med_cov_grid
    Z.set_shape([None,None,gp_params.input_dim]) #somehow lost shape info, but need this
    N = tf.shape(minibatch.T)[0] #number of observations 
    
    tcn_logits = tf.layers.dense(
        tcn(Z, training=minibatch.is_training)[:, -1, :],
        n_classes, activation=None, 
        kernel_initializer=tf.orthogonal_initializer(),
        name='last_linear' 
    ) 

    '''#duplicate each entry of seqlens, to account for multiple MC samples per observation 
    seqlen_dupe = tf.reshape(tf.tile(tf.expand_dims(num_tcn_grid_times,1),[1,gp_params.n_mc_smps]),[N*gp_params.n_mc_smps])

    #with tf.variable_scope("",reuse=True):
    outputs, states = tf.nn.dynamic_tcn(cell=stacked_lstm,inputs=Z,
                                            dtype=tf.float32,
                                            sequence_length=seqlen_dupe)
    
    final_outputs = states[n_layers-1][1]
    preds =  tf.matmul(final_outputs, out_weights) + out_biases  
    '''
    #pass Z forward through tcn:

    return tcn_logits


#old implementation to average the output sequence:
def get_losses(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
              num_tcn_grid_times, cov_grid, input_dim,method, gp_params, tcn, is_training, n_classes, pad_before,
              labels, pos_weight): #med_cov_grid
    """
    helper function. takes in (padded) raw datas, samples MGP for each observation, 
    then feeds it all through the TCN to get predictions.
    
    inputs:
        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
        num_obs_times: number of times observed for each encounter; how long each row of T really is
        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
        num_tcn_grid_times: length of even spaced TCN grid per encounter
                    
    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """
    
    Z = get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                       num_tcn_grid_times, cov_grid, input_dim, method=method, gp_params=gp_params, pad_before=pad_before) #batchsize*num_MC x batch_maxseqlen x num_inputs    ##,med_cov_grid
    Z.set_shape([None,None,input_dim]) #somehow lost shape info, but need this
    N = tf.shape(Z)[0] #number of observations 


    # We only want to consider up tw0 7 timepoints before end
    T_max = 7
    
    # Only during training we want to average over the last few predictions in order to give
    # the model the incentive to predict early
    tcn_out = tcn(Z, training=is_training)[:, -T_max:, :]
    tcn_logits = tf.layers.dense(tcn_out,
        n_classes, activation=None, 
        kernel_initializer=tf.orthogonal_initializer(),
        name='last_linear', reuse=False
    )

    # Only get a few of the last obs
    #used_grid = tf.reduce_min(tf.stack([num_tcn_grid_times, tf.fill(tf.shape(num_tcn_grid_times), T_max)]), axis=0)
    #tiled = tf.tile(tf.expand_dims(used_grid, axis=-1), [1, gp_params.n_mc_smps])
    #expanded_used_grid = tf.reshape(tiled, [-1])
    tiled_labels = tf.tile(tf.expand_dims(labels, axis=1), tf.stack([1, T_max, 1]))
    all_losses = tf.nn.weighted_cross_entropy_with_logits(logits=tcn_logits,targets=tiled_labels, pos_weight=pos_weight)
    average_losses = tf.reduce_mean(all_losses, axis=-1)

    return average_losses

def get_probs_and_accuracy(preds, O, n_mc_smps):
    """
    helper function. we have a prediction for each MC sample of each observation
    in this batch.  need to distill the multiple preds from each MC into a single
    pred for this observation.  also get accuracy. use true probs to get ROC, PR curves in sklearn
    """
    all_probs = tf.exp(preds[:,1] - tf.reduce_logsumexp(preds, axis = 1)) #normalize; and drop a dim so only prob of positive case
    N = tf.cast(tf.shape(preds)[0]/n_mc_smps,tf.int32) #actual number of observations in preds, collapsing MC samples                    
    
    #predicted probability per observation; collapse the MC samples
    probs = tf.zeros([0]) #store all samples in a list, then concat into tensor at end
    #setup tf while loop (have to use this bc loop size is variable)
    def cond(i,probs):
        return i < N
    def body(i,probs):
        probs = tf.concat([probs,[tf.reduce_mean(tf.slice(all_probs,[i*n_mc_smps],[n_mc_smps]))]],0)
        return i+1,probs    
    i = tf.constant(0)
    i,probs = tf.while_loop(cond,body,loop_vars=[i,probs],shape_invariants=[i.get_shape(),tf.TensorShape([None])])
        
    #compare to truth; just use cutoff of 0.5 for right now to get accuracy
    correct_pred = tf.equal(tf.cast(tf.greater(probs,0.5),tf.int32), O)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
    return probs,accuracy


