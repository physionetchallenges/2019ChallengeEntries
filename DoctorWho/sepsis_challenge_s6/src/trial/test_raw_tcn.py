#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Michael Moor October 2018.

"""

from IPython import embed
from sacred import Experiment
from tempfile import NamedTemporaryFile
import traceback
import faulthandler
import os.path 
import pickle
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, '../../src')

#Loading local modules
from utils.memory_saving_gradients import gradients_memory # for memory-saving gradients checking (not a package, but simple source file)
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
tf.__dict__["gradients"] = gradients_memory
from utils.util import count_parameters, get_tcn_window, compute_global_l2, DatasetCompact
from utils.binned_util import pad_binned_data
from tcn.tcn import CausalConv1D, TemporalBlock, TemporalConvNet

from sim_datasets.imputed_datasets import sim_dataset 
from models.raw_tcn import RAWTCN, Minibatch
from dataset.data_loader import extract_data


ex = Experiment('Raw-TCN')

@ex.config
def raw_tcn_config():
    dataset = {
        'datapath': '../../output/',
        'min_pad_length': 8, #None Minimal time series length in padded batches which is max(max_batch_len, min_pad_length) such that no batch is shorter than this parameter (such that TCN works!)
        'split':0 # 0-4
    }
    tcn_parameters = {
        'n_channels': 40,
        'levels': 4, 
        'kernel_size': 4, 
        'dropout': 0.1,
        'n_classes' : 1 
    }
    batch_size = 100 #NOTE may want to play around with this
    learning_rate = 0.001
    training_iters = 30 #num epochs
    L2_penalty = None # using per-weight norm! hence lambda so large.. multiplied with start per weight-norm value arrives at around 10. train loss around 100-2000

@ex.main
def fit_mgp_tcn(batch_size, learning_rate, training_iters, L2_penalty,  _rnd, _seed, _run, dataset, tcn_parameters):
    #Path to save model:
    if len(_run.observers) > 0:
         checkpoint_path = os.path.join(_run.observers[0].dir, 'model_checkpoints')
    else:
        checkpoint_path = 'model_checkpoints'

    rs = np.random.RandomState(_seed)  #using _seed made name error inside dataset sim function
    tf.logging.set_verbosity(tf.logging.ERROR)

    #-----------------
    # Simming dataset:
    #-----------------
     
    num_encs=1000
    M=10
    sim_data = sim_dataset(0, num_encs,M)
    Z, labels = extract_data(sim_data, M) 

    N_tot = len(labels) #total encounters, only interesting when not simming data
    
    #---------------------------------------
    # Splitting dataset into train/val/test:
    #---------------------------------------

    train_test_perm = rs.permutation(N_tot)
    print(f'train test perm, Type: {type(train_test_perm[0]) }')
    val_frac = 0.1 #fraction of full data to set aside for testing
    va_ind = train_test_perm[:int(val_frac*N_tot)]
    tr_ind = train_test_perm[int(val_frac*N_tot):]
    Nva = len(va_ind); Ntr = len(tr_ind)
    
    #Break everything out into train/test
    labels_tr = labels[tr_ind]; labels_va = labels[va_ind]
    Z_tr = Z[tr_ind]; Z_va = Z[va_ind]

    min_pad_length = dataset['min_pad_length'] #minimal time series length a batch should be padded to (for TCNs..)

    #Get class imbalance (for weighted loss):
    #case_prev = labels_tr.sum()/float(len(labels_tr)) #get prevalence of cases in train dataset
    label_sums = [sum(i) for i in labels_tr]
    label_lens = [len(i) for i in labels_tr]
    case_prev = sum(label_sums) / sum(label_lens)

    class_imb = 1/case_prev #class imbalance to use as class weight if losstype='weighted'
    print(f'Using class imbalance of {class_imb}')
 
    #-------------------------
    print("data fully setup!")    
    sys.stdout.flush()
    #-------------------------


    ##test_freq     = Ntr/batch_size #eval on test set after this many batches
    test_freq = int(Ntr/batch_size / 4)
    # Network Parameters
    n_classes = tcn_parameters['n_classes']  #binary outcome 
    input_dim = M

    #Experiment for trying to reproduce randomness..
    tf.set_random_seed(_seed)

    #------------------
    # Start TF Session
    #------------------
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False) #Cave, had to add it to Adam loss min()!
    
    #--------------------------
    # TF Placeholders for Graph
    #--------------------------

    #initialize placeholder minibatch object:
    mb = Minibatch(n_classes)

    #--------------------------
    #Setting up Architecture:
    #--------------------------
    kernel_size, levels = tcn_parameters['kernel_size'], tcn_parameters['levels']
    calculated_length = get_tcn_window(kernel_size, levels)
    if calculated_length > min_pad_length:
        print('Timeseries min_pad_length: {} are too short for Specified TCN Parameters requiring {}'.format(min_pad_length, calculated_length))
        min_pad_length = calculated_length
        print('>>>>>> Setting min_pad_length to: {}'.format(min_pad_length))
    
    #--------------------------
    #Initialize MGP-TCN Model:
    #--------------------------
    model = RAWTCN(tcn_parameters, input_dim, class_imb, L2_penalty)   
    
    #Define model operations: 
    preds = model.predict(mb) #model takes minibatch object and returns predictions
    ### probs, accuracy = model.probs_and_accuracy(preds, mb)
    loss = model.loss(preds, mb)
 
    #Define training operation
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step) ## added global step to minimize()
   
    # Initialize globals and get ready to start!
    sess.run(tf.global_variables_initializer())
    print("Graph setup!")
    count_parameters()

    ### Add runoptions for memory issues:
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

    #setup minibatch indices for training:
    starts = np.arange(0,Ntr,batch_size)
    ends = np.arange(batch_size,Ntr+1,batch_size)
    if ends[-1]<Ntr: 
        ends = np.append(ends,Ntr)
    num_batches = len(ends)

    #setup minibatch indices for validation (memory saving)
    va_starts = np.arange(0,Nva,batch_size)
    va_ends = np.arange(batch_size,Nva+1,batch_size)
    if va_ends[-1]<Nva: 
        va_ends = np.append(va_ends,Nva)
 
    #here: initial position of validation padding..
    
    Z_pad_va, labels_pad_va, valid_points_va = pad_binned_data(Z_va, labels_va, min_pad_length)
    
    #------------------- 
    # Main training loop
    #-------------------
    saver = tf.train.Saver(max_to_keep = None)

    total_batches = 0
    best_val = 0
    for i in range(training_iters):
        print('Max Memory Usage up to now')
        print(sess.run(tf.contrib.memory_stats.MaxBytesInUse()))
        
        #------------------- 
        # Train
        #------------------- 
        epoch_start = time()
        print("Starting epoch "+"{:d}".format(i))
        perm = rs.permutation(Ntr)
        batch = 0 
        for s,e in zip(starts,ends):

            batch_start = time()
            inds = perm[s:e]
            Z_pad_tr, labels_pad_tr, valid_points_tr = pad_binned_data(Z_tr[inds], labels_tr[inds], min_pad_length)
            #TODO:
            # - adjust feed dict (still need to do it for train data)
            # X adjust RAWTCN class to return logits for full time series
            # - compute loss for full tensor but select valid points cross_entr returns same shaped tensor as labels or logits!       
     
            feed_dict={ mb.Z: Z_pad_tr, 
                        mb.O: labels_pad_tr,
                        mb.valid_points: valid_points_tr,
                        mb.is_training: True
                        }
        
            #feed_dict={mb.Y:Y_pad,mb.T:T_pad,mb.ind_kf:ind_kf_pad,mb.ind_kt:ind_kt_pad,
            #    mb.X:X_pad, mb.num_obs_times:num_obs_times_tr[inds],
            #    mb.num_obs_values:num_obs_values_tr[inds],
            #    mb.num_tcn_grid_times:num_tcn_grid_times_tr[inds],
            #    mb.O:labels_tr[inds], mb.is_training: True}
 
            try:        
                loss_,_ = sess.run([loss,train_op],feed_dict, options=run_options)
                
            except Exception as e:
                traceback.format_exc()
                print('Error occured in tensorflow during training:', e)
                #In addition dump more detailed traceback to txt file:
                with NamedTemporaryFile(suffix='.csv') as f:
                    faulthandler.dump_traceback(f)
                    _run.add_artifact(f.name, 'faulthandler_dump.csv')
                break
            
            print("Batch "+"{:d}".format(batch)+"/"+"{:d}".format(num_batches)+\
                  ", took: "+"{:.3f}".format(time()-batch_start)+", loss: "+"{:.5f}".format(loss_))
            sys.stdout.flush()
            batch += 1; total_batches += 1

            if total_batches % test_freq == 0: #Check val set every so often for early stopping

                #--------------------------
                # Validate and write scores
                #-------------------------- 

                print('--> Entering validation step...')
                #TODO: may also want to check validation performance at additional X hours back
                #from the event time, as well as just checking performance at terminal time
                #on the val set, so you know if it generalizes well further back in time as well 
                val_t = time()
                #Batch-wise Validation Phase:
                va_probs_tot = np.array([])
                va_perm = rs.permutation(Nva)
                va_labels_tot = labels_va[va_perm]
                for v_s,v_e in zip(va_starts,va_ends):
                    va_inds = va_perm[v_s:v_e]
                    
                    va_feed_dict={ mb.Z: Z_pad_va[va_inds], 
                        mb.O: labels_pad_va[va_inds],
                        mb.valid_points: valid_points_va[va_inds],
                        mb.is_training: False
                        }

                    #try:        
                    va_loss = sess.run([loss],va_feed_dict, options=run_options)  
                    #except Exception as e:
                    #    traceback.format_exc()
                    #    print('Error occured in tensorflow during evaluation:', e)
                    #    break
                    #append current validation auprc to array of entire validation set
                    ### va_probs_tot = np.concatenate([va_probs_tot, va_probs])

                #print('Type of va_probs: {} and labels_va'.format(type(va_probs), type(labels_va)))
                #print('Shape of va_probs: {} and labels_va'.format(va_probs.shape, labels_va.shape))
                #assuming both come as np array:


                ### va_auc = roc_auc_score(va_labels_tot, va_probs_tot)
                ### va_prc = average_precision_score(va_labels_tot, va_probs_tot)   
                ### best_val = max(va_prc, best_val)
                print(f"Epoch {i}, seen {total_batches} total batches. Validating Took " +\
                      f"{time()-val_t} seconds. " +\
                      f" Loss: {va_loss}" ) ###+ \
                      ###", AUC: {:.5f}".format(va_auc)+", AUPR: "+"{:.5f}".format(va_prc))
                _run.log_scalar('train_loss', loss_, total_batches)
                ### _run.log_scalar('val_auprc', va_prc, total_batches)
                sys.stdout.flush()    
            
                #create a folder and put model checkpoints there
                saver.save(sess, checkpoint_path + "/epoch_{}".format(i), global_step=total_batches)
        print("Finishing epoch "+"{:d}".format(i)+", took "+\
              "{:.3f}".format(time()-epoch_start))     
        
    ### return {'Best Validation AUPRC': best_val}

if __name__ == '__main__':
    ex.run_commandline()
