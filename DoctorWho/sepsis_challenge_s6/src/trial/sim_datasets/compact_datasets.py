'''
Script that contains dataset code:
- simulation dataset
- 
'''
import tensorflow as tf
import numpy as np

#---------------------------------
# 1. CODE FOR SIMULATION DATASET:
#---------------------------------

#Sim dataset functions from mgp_rnn_fit.py (jfutoma)
def gen_MGP_params(M, rs):
    """
    Generate some MGP params for each class. 
    Assume MGP is stationary and zero mean, so hyperparams are just:
        Kf: covariance across time series
        length: length scale for shared kernel across time series
        noise: noise level for each time series
    """ 
    
    true_Kfs = []
    true_noises = []
    true_lengths = []
    
    #Class 0
    tmp = rs.normal(0,.2,(M,M))
    true_Kfs.append(np.dot(tmp,tmp.T))
    true_lengths.append(1.0)
    true_noises.append(np.linspace(.02,.08,M))
    
    #Class 1
    tmp = rs.normal(0,.4,(M,M))
    true_Kfs.append(np.dot(tmp,tmp.T))
    true_lengths.append(2.0)
    true_noises.append(np.linspace(.1,.2,M))
    
    return true_Kfs,true_noises,true_lengths
    
    
def sim_dataset(rs, num_encs,M,n_covs,n_meds,pos_class_rate = 0.5,trainfrac=0.2):
    """
    Returns everything we need to run the model.
    
    Each simulated patient encounter consists of:
        Multivariate time series (labs/vitals)
        Static baseline covariates
        Medication administration times (no values; just a point process)
    """
    true_Kfs,true_noises,true_lengths = gen_MGP_params(M, rs)
        
    end_times = np.random.uniform(10,50,num_encs) #last observation time of the encounter
    num_obs_times = np.random.poisson(end_times,num_encs)+3 #number of observation time points per encounter, increase with  longer series 
    num_obs_values = np.array(num_obs_times*M*trainfrac,dtype="int")
    #number of inputs to RNN. will be a grid on integers, starting at 0 and ending at next integer after end_time
    num_rnn_grid_times = np.array(np.floor(end_times)+1,dtype="int") 
    rnn_grid_times = []
    labels = rs.binomial(1,pos_class_rate,num_encs)                      
    
    T = [];  #actual observation times
    Y = []; ind_kf = []; ind_kt = [] #actual data; indices pointing to which lab, which time
    baseline_covs = np.zeros((num_encs,n_covs)) 
    #each contains an array of size num_rnn_grid_times x n_meds 
    #   simulates a matrix of indicators, where each tells which meds have been given between the
    #   previous grid time and the current.  in practice you will have actual medication administration 
    #   times and will need to convert to this form, for feeding into the RNN
    meds_on_grid = [] 

    print('Simming data...')
    for i in range(num_encs):
        if i%500==0:
            print('%d/%d' %(i,num_encs))
        obs_times = np.insert(np.sort(np.random.uniform(0,end_times[i],num_obs_times[i]-1)),0,0)
        T.append(obs_times)
        l = labels[i]
        y_i,ind_kf_i,ind_kt_i = sim_multitask_GP(obs_times,true_lengths[l],true_noises[l],true_Kfs[l],trainfrac)
        Y.append(y_i); ind_kf.append(ind_kf_i); ind_kt.append(ind_kt_i)
        rnn_grid_times.append(np.arange(num_rnn_grid_times[i]))
        if l==0: #sim some different baseline covs; meds for 2 classes
            baseline_covs[i,:int(n_covs/2)] = rs.normal(0.1,1.0,int(n_covs/2))
            baseline_covs[i,int(n_covs/2):] = rs.binomial(1,0.2,int(n_covs/2))
            meds = rs.binomial(1,.02,(num_rnn_grid_times[i],n_meds))
        else:
            baseline_covs[i,:int(n_covs/2)] = rs.normal(0.2,1.0,int(n_covs/2))
            baseline_covs[i,int(n_covs/2):] = rs.binomial(1,0.1,int(n_covs/2))
            meds = rs.binomial(1,.04,(num_rnn_grid_times[i],n_meds))
        meds_on_grid.append(meds)
    
    T = np.array(T)
    Y = np.array(Y); ind_kf = np.array(ind_kf); ind_kt = np.array(ind_kt)
    meds_on_grid = np.array(meds_on_grid)
    rnn_grid_times = np.array(rnn_grid_times)
    
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,
            labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs)
    
def sim_multitask_GP(times,length,noise_vars,K_f,trainfrac):
    """
    draw from a multitask GP.  
    
    we continue to assume for now that the dim of the input space is 1, ie just time

    M: number of tasks (labs/vitals/time series)
    
    train_frac: proportion of full M x N data matrix Y to include

    """
    M = np.shape(K_f)[0]
    N = len(times)
    n = N*M
    K_t = OU_kernel_np(length,times) #just a correlation function
    Sigma = np.diag(noise_vars)

    K = np.kron(K_f,K_t) + np.kron(Sigma,np.eye(N)) + 1e-6*np.eye(n)
    L_K = np.linalg.cholesky(K)
    
    y = np.dot(L_K,np.random.normal(0,1,n)) #Draw normal
    
    #get indices of which time series and which time point, for each element in y
    ind_kf = np.tile(np.arange(M),(N,1)).flatten('F') #vec by column
    ind_kx = np.tile(np.arange(N),(M,1)).flatten()
               
    #randomly dropout some fraction of fully observed time series
    perm = np.random.permutation(n)
    n_train = int(trainfrac*n)
    train_inds = perm[:n_train]
    
    y_ = y[train_inds]
    ind_kf_ = ind_kf[train_inds]
    ind_kx_ = ind_kx[train_inds]
    
    return y_,ind_kf_,ind_kx_

def OU_kernel_np(length,x):
    """ just a correlation function, for identifiability 
    """
    x1 = np.reshape(x,[-1,1]) #colvec
    x2 = np.reshape(x,[1,-1]) #rowvec
    K_xx = np.exp(-np.abs(x1-x2)/length)    
    return K_xx






