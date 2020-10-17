#!/usr/bin/env python

import numpy as np

def get_sepsis_score(data, model):
    feature_matrix = data
    feature_matrix[np.isnan(feature_matrix)]=-1

    # Use model parameters
    ESNtools = model['f']
    
    ## ESN Generation parameters
    N = model['N_def']                        # Neurons
    mem = model['mem_def']                    # memory
    scale = model['scale_def']                # scaling factor

    ## Nonlinear mapping function
    sigmoid_exponent = model['exponent_def']  # sig exponent
    func = ESNtools.sigmoid
    
    ## Mask parameters
    # M = 2*np.random.rand(np.shape(feature_matrix)[1],N)-1
    # Mb = 2*np.random.rand(1,N)-1
    M = model['M']
    Mb = model['Mb']    


    ##Weights and thresholds
    w = model['w']
    th_max = model['th_max'] 
    th_min = model['th_min']
    th_scale = model['th_scale']

    ## Perform ESN feed
    ESN = ESNtools.feedESN(feature_matrix, N, M, Mb, scale, mem, func, sigmoid_exponent)
    del feature_matrix
    
    
    ## Compute class prediction
    single_sample = True
    if single_sample:
        Y_pred = (np.matmul(ESN[-1,:],w))
        scores = (Y_pred - th_min) / th_scale
        labels = np.asarray(Y_pred > th_max, dtype = np.int)
        if scores > 1.0 :
            scores = 1.0
        elif scores < 0.0 : 
            scores = 0.0

    else:
        Y_pred = (np.matmul(ESN,w))
        scores = (Y_pred - th_min) / th_scale
        labels = np.asarray(Y_pred > th_max, dtype = np.int)
        scores[np.where(scores > 1.0)[0]]=1.0
        scores[np.where(scores < 0.0)[0]]=0.0
        print(scores)
        print(np.shape(scores))
    return scores, labels

def load_sepsis_model():
    import scipy.linalg as linalg
    
    # Random seed
    np.random.seed(seed=0)
    class ESNT:
        """
        ESN tools module
        """
        
        ### Map data ################################################################
        @staticmethod
        def sigmoid(x, exponent):
            """Apply a [-0.5, 0.5] sigmoid function."""
            
            return 1/(1+np.exp(-exponent*x))-0.5
        
        ### Feed data into Echo State Network #######################################
        @staticmethod
        def feedESN(features, neurons, mask, mask_bias, scale, mem, func, f_arg):
            """Feeds data into a ring Echo State Network. Returns ESN state.
            Adds extra (1) neuron for Ax + b = Y linear system.
        
            Parameters
            ----------
            features : (np.array) feature matrix original data (samples,features)
            
            neurons : (int) number of neurons to use
        
            mask : (np.array) input weights mask matrix (usually randomly generated)
            
            mask_bias : (np.array) initialisation bias per neuron
            
            scale : (float) input scaling factor
        
            mem : (float) memory feedback factor
        
            func : (function) nonlinear mapping function
        
            f_arg : (float) function parameter. sigmoid exponent or slope in rect
            """
            
            ESN = np.hstack((np.matmul(features, mask), np.ones((np.shape(features)[0],1), dtype=np.double)))
            p = np.zeros((1,neurons),dtype=np.double)
        
            for i in range(np.shape(features)[0]):
                in_val = scale * (ESN[i,:-1] + mask_bias) + p * mem
                
                ## Apply transform
                ESN[i,:-1] = func(in_val, f_arg)
                
                ## Connect preceding neighbour 
                p = np.copy(np.roll(ESN[i,:-1],1))
            return ESN
        
        ### Get ESN training weights (NE: normal eq.) ############################
        @staticmethod
        def get_weights_biasedNE(ESN, target):
            """Computes ESN training weights solving (pinv) the NE linear system w/ bias.
            Parameters
            ----------
            ESN : (np.array) Echo State Network state
            
            target : (np.array) target labels to train with
            
            """
            Y_aux = np.matmul(ESN.T,target)
            ESNinv = np.linalg.pinv(np.matmul(ESN.T,ESN))
            w = np.matmul(ESNinv, Y_aux)
            return w
        
        @staticmethod
        def get_weights_qr_biasedNE(ESN, target):
            """Computes ESN training weights solving (qr) the NE linear system w/ bias.
            Parameters
            ----------
            ESN : (np.array) Echo State Network state
            
            target : (np.array) target labels to train with
            
            """
            Q, R = linalg.qr((np.matmul(ESN.T,ESN)))            # QR decomposition with qr function (RtR)w = RtY
            Y_aux = np.dot(Q.T, np.matmul(ESN.T, target))       # Let y=Q'.ESNt using matrix multiplication
            w = linalg.solve(R, Y_aux)                          # Solve Rx=y
            return w
        
        @staticmethod
        def get_weights_lu_biasedNE(ESN, target):
            """Computes ESN training weights solving (lu) the NE linear system w/ bias.
            Parameters
            ----------
            ESN : (np.array) Echo State Network state
            
            target : (np.array) target labels to train with
            
            """
            LU = linalg.lu_factor((np.matmul(ESN.T,ESN)))      # LU decomposition with (RtR)w = RtY
            Y_aux = (np.matmul(ESN.T, target))                 # Let Y=ESNt.y using matrix multiplication
            w = linalg.lu_solve(LU, Y_aux)                     # Solve Rx=y
            return w

    esnt = ESNT()
    model = dict()
    with open('w.txt') as file:
        w = (np.loadtxt(file, skiprows=1))
        
    # Model parameters
    model['N_def'] = 100         # Neurons
    model['scale_def'] = 0.001   # scaling
    model['mem_def'] = 0.1       # memory
    model['exponent_def'] = 1.0  # sigmoid exponent

    # Thresholds
    model['th_max'] = 0.0559 
    model['th_min'] = -0.9345
    model['th_scale'] = 1.8581
    
    # Model functions
    model['f'] = esnt
    model['type'] = 'ESN'
    model['w'] = w

    # Model Mask
    model['M'] = 2*np.random.rand(40,model['N_def'])-1
    model['Mb'] = 2*np.random.rand(1,model['N_def'])-1

    return model
