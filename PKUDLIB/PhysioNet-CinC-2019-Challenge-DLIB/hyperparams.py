# -*- coding: utf-8 -*-
'''
April 2019 by WHX
Modified from
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import numpy as np

class Hyperparams_selfi:
    batch_size = 64
    learning_rate = 0.00003
    logdir = 'SAnD_bce_selfi2' # log directory
    # model
    timelen = 208
    event_nums = 40         
    hidden_units = 64
    num_blocks = 4 
    num_epochs = 400
    num_heads = 8
    dropout_rate = 0.3
    mask = np.ones((timelen, timelen), dtype = np.bool)
    for i in range(timelen):
        for j in range(max(0, i - 48),i + 1):
            mask[i][j] = False

class Hyperparams_augselfi:
    batch_size = 128 
    learning_rate = 0.0001
    logdir = 'SAnD_bce_augselfi' # log directory
    # model
    timelen = 208
    event_nums = 40         
    hidden_units = 16
    num_blocks = 2 
    num_epochs = 400
    num_heads = 4
    dropout_rate = 0.3
    DI_M = 12
    mask = np.ones((timelen, timelen), dtype = np.bool)
    for i in range(timelen):
        for j in range(0, i + 1):
            mask[i][j] = False
    
class Hyperparams_s2s:
    batch_size = 128 
    learning_rate = 0.0001
    logdir = 'SAnD_bce_s2s' # log directory
    # model
    timelen = 336
    event_nums = 40         
    hidden_units = 16
    num_blocks = 1 
    num_epochs = 400
    num_heads = 4
    dropout_rate = 0.3
    r = 336
    mask = np.ones((timelen, timelen), dtype = np.bool)
    for i in range(timelen):
        for j in range(max(0, i - r + 1), i + 1):
            mask[i][j] = False

class Hyperparams_reg:
    batch_size = 128 
    learning_rate = 0.0001
    logdir = 'SAnD_bce_reg' # log directory
    # model
    timelen = 208
    event_nums = 40         
    hidden_units = 16
    num_blocks = 2 
    num_epochs = 400
    num_heads = 4
    dropout_rate = 0.3
    DI_M = 12
    mask = np.ones((timelen, timelen), dtype = np.bool)
    for i in range(timelen):
        for j in range(0, i + 1):
            mask[i][j] = False

class Hyperparams_LSTM:
    batch_size = 256 
    learning_rate = 0.0001
    logdir = 'LSTM' # log directory
    # model
    timelen = 48
    event_nums = 40         
    hidden_units = 16
    num_epochs = 300
    dropout_rate = 0.3

class Hyperparams_utf:
    batch_size = 256 
    learning_rate = 0.0003
    logdir = 'SAnD_bce_utf2' # log directory
    # model
    timelen = 208
    event_nums = 40  
    max_stacks = 4
    act_epsilon = 0.01       
    hidden_units = 32
    num_epochs = 400
    num_heads = 4
    dropout_rate = 0.3
    DI_M = 208
    mask = np.ones((timelen, timelen), dtype = np.bool)
    for i in range(timelen):
        for j in range(0, i + 1):
            mask[i][j] = False

class Hyperparams_decom:
    batch_size = 256
    learning_rate = 0.00005
    logdir = 'SAnD_bce_decom_hembed' # log directory
    # model
    timelen = 24
    event_nums = 40        
    hidden_units = 8
    num_blocks = 1
    num_epochs = 400
    num_heads = 2
    pool_heads = 4
    dropout_rate = 0.3
    DI_M = 2
    l2r = 0.05
    class_weighted = 1

class Hyperparams_dos:
    k = 10
    epoch_nums = 200
    batch_size = 256
    learning_rate = 0.0001
    logdir = 'SAnD_dos' # log directory
    # model
    timelen = 24
    event_nums = 43  
    hidden_units = 16
    num_blocks = 2
    num_epochs = 400
    num_heads = 2
    dropout_rate = 0.3
    DI_M = 2
    flr = 0.05
    r = 6 