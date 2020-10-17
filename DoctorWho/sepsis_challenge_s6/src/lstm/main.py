#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Michael Moor October 2018.

"""

from IPython import embed
from sacred import Experiment
from sacred.observers import FileStorageObserver
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
sys.path.insert(0, '../../scripts')

#Loading local modules
from utils.memory_saving_gradients import gradients_memory # for memory-saving gradients checking (not a package, but simple source file)
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
tf.__dict__["gradients"] = gradients_memory
from utils.util import count_parameters, get_tcn_window, get_pointwise_preds_and_probs 
from utils.binned_util import pad_binned_data

from models.lstm import LSTM_Model, Minibatch
from dataset.data_loader import extract_data, data_loader, load_and_extract_splits, compute_prevalence
from evaluation.evaluate_sepsis_score import custom_scoring

#from scripts:
from plot_metric import plot_metric

ex = Experiment('LSTM')
ex.observers.append(FileStorageObserver.create('lstm_runs'))

@ex.config
def raw_tcn_config():
    dataset = {
        'datapath': '../../../../data/mgp_tcn_data/',
        'min_pad_length': 8, #None Minimal time series length in padded batches which is max(max_batch_len, min_pad_length) such that no batch is shorter than this parameter (such that TCN works!)
        'split': 0, # 0-4
        'indicator': True
    }
    tcn_parameters = {
        'n_channels': 20,
        'levels': 3, 
        'dropout': None, #rnn dropout (with 1.0 as default) not implemented yet! #0.0 for tcn dropout
        'n_classes' : 1,
        'class_imbalance_parameter' : 0.5 
    }
    batch_size = 64 #NOTE may want to play around with this
    learning_rate = 0.001
    training_iters = 50 #num epochs
    L2_penalty = None # using per-weight norm! hence lambda so large.. multiplied with start per weight-norm value arrives at around 10. train loss around 100-2000

@ex.main
def fit_lstm(batch_size, learning_rate, training_iters, L2_penalty,  _rnd, _seed, _run, dataset, tcn_parameters):
    #Path to save model:
    if len(_run.observers) > 0:
         checkpoint_path = os.path.join(_run.observers[0].dir, 'model_checkpoints')
    else:
        checkpoint_path = 'model_checkpoints'

    rs = np.random.RandomState(_seed)  #using _seed made name error inside dataset sim function
    tf.logging.set_verbosity(tf.logging.ERROR)

    #-----------------------------
    # Loading dataset into splits:
    #-----------------------------
    split = dataset['split']
    if dataset['indicator']:
        filename = f'imputed_indicated_data_split_{split}.pkl'
    else:
        filename = f'imputed_data_split_{split}.pkl'
    filepath = dataset['datapath'] + filename 
    data = load_and_extract_splits(filepath)
  
    ##Break everything out into train/test
    labels_tr = data['train']['y']
    labels_va = data['validation']['y']
    Z_tr = data['train']['X']
    Z_va = data['validation']['X']
    ids_tr = data['train']['pat_ids']
    ids_va = data['validation']['pat_ids']
    #print(f'Train-IDs: {ids_tr}')

    Nva = len(labels_va); Ntr = len(labels_tr)
    min_pad_length = dataset['min_pad_length'] #minimal time series length a batch should be padded to (for TCNs..)

    #Get class imbalance (for weighted loss):
    case_prev = compute_prevalence(labels_tr)
    class_imbalance_parameter = tcn_parameters['class_imbalance_parameter']
    if class_imbalance_parameter is None:
        class_imb = 1
    else:
        class_imb = class_imbalance_parameter / case_prev #class imbalance to use as class weight if losstype='weighted'
    print(f'Using class imbalance of {class_imb}, and case prevalence of {case_prev}')
 
    #-------------------------
    print("data fully setup!")    
    sys.stdout.flush()
    #-------------------------

    test_freq = int(Ntr/batch_size / 4) #eval 4 times per epoch
    # Network Parameters
    n_classes = tcn_parameters['n_classes']  #binary outcome 
    input_dim = data['train']['X'][0].shape[1]
    print(f'input dim = {input_dim}')
 
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
    #Initialize MGP-TCN Model:
    #--------------------------
    model = LSTM_Model(tcn_parameters, input_dim, class_imb, L2_penalty)   
    
    #Define model operations: 
    preds = model.predict(mb) #model takes minibatch object and returns predictions
    probs, mask = model.probs(preds, mb) #get probabilities (padded) and mask indicating valid indices)
    loss = model.loss(preds, mb)
 
    saver = tf.train.Saver(max_to_keep = None)

    #Define training operation
    
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
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
    
    #Z_pad_va, labels_pad_va, valid_points_va = pad_binned_data(Z_va, labels_va, min_pad_length)
    
    #------------------- 
    # Main training loop
    #-------------------

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
            #print(f'Z pad shape: {Z_pad_tr.shape}')
            #print(f'labels pad shape: {labels_pad_tr.shape}')
 
            #TODO:
            # - adjust feed dict (still need to do it for train data)
            # X adjust RAWTCN class to return logits for full time series
            # - compute loss for full tensor but select valid points cross_entr returns same shaped tensor as labels or logits!       
     
            feed_dict={ mb.Z: Z_pad_tr, 
                        mb.O: labels_pad_tr,
                        mb.valid_points: valid_points_tr,
                        mb.is_training: True
                        }
        
 
            loss_,_, tr_probs, tr_mask = sess.run([loss, train_op, probs, mask],feed_dict, options=run_options)
            #except Exception as e:
            #    traceback.format_exc()
            #    print('Error occured in tensorflow during training:', e)
            #    #In addition dump more detailed traceback to txt file:
            #    with NamedTemporaryFile(suffix='.csv') as f:
            #        faulthandler.dump_traceback(f)
            #        _run.add_artifact(f.name, 'faulthandler_dump.csv')
            #    break
            
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
                #initialise arrays to gather batch-wise outputs:
                va_probs_tot = [] #np.array([])
                va_mask_tot = [] #np.array([])
                #Set up minibatch-wise validation:
                va_perm = rs.permutation(Nva)
                va_labels_tot = labels_va[va_perm]
                for v_s,v_e in zip(va_starts,va_ends):
                    va_inds = va_perm[v_s:v_e]
                    
                    Z_pad_va, labels_pad_va, valid_points_va = pad_binned_data(Z_va[va_inds], labels_va[va_inds], min_pad_length)

                    va_feed_dict={ mb.Z: Z_pad_va, 
                        mb.O: labels_pad_va,
                        mb.valid_points: valid_points_va,
                        mb.is_training: False
                        }

                    #try:        
                    va_loss, va_probs, va_mask = sess.run([loss, probs, mask],va_feed_dict, options=run_options)  
                    #except Exception as e:
                    #    traceback.format_exc()
                    #    print('Error occured in tensorflow during evaluation:', e)
                    #    break
                    #append current validation auprc to array of entire validation set
                    va_probs_tot.append(va_probs) #= np.concatenate([va_probs_tot, va_probs]) #va_probs_tot.append(va_probs) #
                    va_mask_tot.append(va_mask) # = np.concatenate([va_mask_tot, va_mask]) #va_mask_tot.append(va_mask) #

                #print('Type of va_probs: {} and labels_va'.format(type(va_probs), type(labels_va)))
                #print('Shape of va_probs: {} and labels_va'.format(va_probs.shape, labels_va.shape))
                #assuming both come as np array:
                
                #get pat ids of current validation perm:
                va_ids_total = ids_va[va_perm]
                pw_preds, pw_probs = get_pointwise_preds_and_probs(va_probs_tot, va_mask_tot )
                print(f'Labels len: {len(va_labels_tot)}')
                print(f'Predictions len: {len(pw_preds)}')
                print(f'Probs len: {len(pw_probs)}')
                va_auc, va_prc, va_util = custom_scoring(pw_preds, pw_probs, va_labels_tot) 
                ### va_auc = roc_auc_score(va_labels_tot, va_probs_tot)
                ### va_prc = average_precision_score(va_labels_tot, va_probs_tot)   
                ### best_val = max(va_prc, best_val)
                print(f"Epoch {i}, seen {total_batches} total batches. Validating Took " +\
                      f"{time()-val_t} seconds. " +\
                      f" Loss: {va_loss}" + \
                      ", AUC: {:.5f}".format(va_auc)+", AUPR: {:.5f}".format(va_prc) +\
                      ", UTILIY: {:.5f}".format(va_util)
                      )
                _run.log_scalar('train_loss', loss_, total_batches)
                _run.log_scalar('validation_loss', va_loss, total_batches)
                _run.log_scalar('validation_AUC', va_auc, total_batches)
                _run.log_scalar('validation_AUPRC', va_prc, total_batches)
                _run.log_scalar('validation_UTILIY', va_util, total_batches)
                ### _run.log_scalar('val_auprc', va_prc, total_batches)
                sys.stdout.flush()    
            
                #create a folder and put model checkpoints there
                saver.save(sess, checkpoint_path + "/epoch_{}".format(i), global_step=total_batches)
        print("Finishing epoch "+"{:d}".format(i)+", took "+\
              "{:.3f}".format(time()-epoch_start))     
        
    ### return {'Best Validation AUPRC': best_val}
    

    return _run._id  


 
if __name__ == '__main__':
    run_id_obj = ex.run_commandline()
    run_id = run_id_obj.result

    plot_title = f'../../plots/lstm_run_{run_id}'  
    plot_data_path = f'lstm_runs/{run_id}/metrics.json' 
    plot_metric(plot_title, plot_data_path)
