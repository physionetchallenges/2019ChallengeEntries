'''
MGP Module (code for the most part copied from Futoma et al. 2017 ICML)
'''
import tensorflow as tf
import numpy as np
from .mgp_utils import OU_kernel,CG,Lanczos,block_CG,block_Lanczos

#------------------------------------------------
##### Convinience classes for managing parameters


class DecompositionMethod():
    valid_methods = ['chol', 'cg']
    def __init__(self, methodname, add_diag=1e-3):
        if methodname not in self.valid_methods:
            raise ValueError('{} is not a valid methodname. Must be one of {}'.format(methodname, self.valid_methods))
        self.methodname = methodname
        self.add_diag = add_diag

class GPParameters():
    def __init__(self, input_dim, n_mc_smps, decomp_method, pad_before):
        self.input_dim = input_dim
        self.log_length = tf.Variable(tf.random_normal([1],mean=1,stddev=0.1),name="GP-log-length") 
        self.length = tf.exp(self.log_length)

        #different noise level of each lab
        self.log_noises = tf.Variable(tf.random_normal([input_dim],mean=-2,stddev=0.1),name="GP-log-noises")
        self.noises = tf.exp(self.log_noises)

        #init cov between labs
        self.L_f_init = tf.Variable(tf.eye(input_dim),name="GP-Lf")
        self.Lf = tf.matrix_band_part(self.L_f_init,-1,0)
        self.Kf = tf.matmul(self.Lf,tf.transpose(self.Lf))

        self.n_mc_smps = n_mc_smps

        #which decomposition method of Covariance matrix to use:
        self.method = decomp_method
        
        #boolean if GP draws should be padded before or afterwards (in time axis) before clf
        self.pad_before = pad_before

#------------------------------------------------
##### Tensorflow functions to draw samples from MGP

def draw_GP(Yi,Ti,Xi,ind_kfi,ind_kti, gp_params):
    """ 
    given GP hyperparams and data values at observation times, draw from 
    conditional GP
    
    inputs:
        length,noises,Lf,Kf: GP params
        Yi: observation values
        Ti: observation times
        Xi: grid points (new times for tcn)
        ind_kfi,ind_kti: indices into Y
    returns:
        draws from the GP at the evenly spaced grid times Xi, given hyperparams and data
    """  
    n_mc_smps, length, noises, Lf, Kf, method = gp_params.n_mc_smps, gp_params.length, gp_params.noises, gp_params.Lf, gp_params.Kf, gp_params.method
    M = gp_params.input_dim
    ny = tf.shape(Yi)[0]
    K_tt = OU_kernel(length,Ti,Ti)
    D = tf.diag(noises)

    grid_f = tf.meshgrid(ind_kfi,ind_kfi) #same as np.meshgrid
    Kf_big = tf.gather_nd(Kf,tf.stack((grid_f[0],grid_f[1]),-1))
    
    grid_t = tf.meshgrid(ind_kti,ind_kti) 
    Kt_big = tf.gather_nd(K_tt,tf.stack((grid_t[0],grid_t[1]),-1))

    Kf_Ktt = tf.multiply(Kf_big,Kt_big)

    DI_big = tf.gather_nd(D,tf.stack((grid_f[0],grid_f[1]),-1))
    DI = tf.diag(tf.diag_part(DI_big)) #D kron I
    
    #data covariance. 
    #Either need to take Cholesky of this or use CG / block CG for matrix-vector products
    Ky = Kf_Ktt + DI + method.add_diag*tf.eye(ny)   

    ### build out cross-covariances and covariance at grid
    
    nx = tf.shape(Xi)[0]
    
    K_xx = OU_kernel(length,Xi,Xi)
    K_xt = OU_kernel(length,Xi,Ti)
                       
    ind = tf.concat([tf.tile([i],[nx]) for i in range(M)],0)
    grid = tf.meshgrid(ind,ind)
    Kf_big = tf.gather_nd(Kf,tf.stack((grid[0],grid[1]),-1))
    ind2 = tf.tile(tf.range(nx),[M])
    grid2 = tf.meshgrid(ind2,ind2)
    Kxx_big =  tf.gather_nd(K_xx,tf.stack((grid2[0],grid2[1]),-1))
    
    K_ff = tf.multiply(Kf_big,Kxx_big) #cov at grid points           
                 
    full_f = tf.concat([tf.tile([i],[nx]) for i in range(M)],0)            
    grid_1 = tf.meshgrid(full_f,ind_kfi,indexing='ij')
    Kf_big = tf.gather_nd(Kf,tf.stack((grid_1[0],grid_1[1]),-1))
    full_x = tf.tile(tf.range(nx),[M])
    grid_2 = tf.meshgrid(full_x,ind_kti,indexing='ij')
    Kxt_big = tf.gather_nd(K_xt,tf.stack((grid_2[0],grid_2[1]),-1))

    K_fy = tf.multiply(Kf_big,Kxt_big)
       
    #now get draws!
    y_ = tf.reshape(Yi,[-1,1])
    
    xi = tf.random_normal((nx*M, n_mc_smps))
    #print('xi shape:')
    #print(xi.shape)
    
    if method.methodname == 'chol':
        Ly = tf.cholesky(Ky)
        Mu = tf.matmul(K_fy,tf.cholesky_solve(Ly,y_))
        Sigma = K_ff - tf.matmul(K_fy,tf.cholesky_solve(Ly,tf.transpose(K_fy))) + method.add_diag*tf.eye(tf.shape(K_ff)[0]) 
        #Exp2: increase noise on Sigma 1e-6 to 1e-3, to 1e-1?
        #Sigma = tf.cast(Sigma, tf.float64) ## Experiment: is chol instable and needs float64? Will this crash Memory?
        #draw = Mu + tf.matmul(tf.cast(tf.cholesky(Sigma),tf.float32),xi) 
        draw = Mu + tf.matmul(tf.cholesky(Sigma),xi) 
        draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),[n_mc_smps,M,nx]),perm=[0,2,1])
        #print('cholesky draw:')
        #print(sess.run(draw_reshape))

    elif method.methodname == 'cg':
        Mu = tf.matmul(K_fy,CG(Ky,y_)) #May be faster with CG for large problems
        #Never need to explicitly compute Sigma! Just need matrix products with Sigma in Lanczos algorithm
        def Sigma_mul(vec):
            # vec must be a 2d tensor, shape (?,?) 
            return tf.matmul(K_ff,vec) - tf.matmul(K_fy,block_CG(Ky,tf.matmul(tf.transpose(K_fy),vec))) 
        def large_draw():             
            return Mu + block_Lanczos(Sigma_mul,xi,n_mc_smps) #no need to explicitly reshape Mu
        #draw = tf.cond(tf.less(nx*M,BLOCK_LANC_THRESH),small_draw,large_draw)
        draw = large_draw()
        draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),[n_mc_smps,M,nx]),perm=[0,2,1])
        #print('cg draw shape:')
        #print(draw_reshape.shape)   

    #TODO: it's worth testing to see at what point computation speedup of Lanczos algorithm is useful & needed.
    # For smaller examples, using Cholesky will probably be faster than this unoptimized Lanczos implementation.
    # Likewise for CG and BCG vs just taking the Cholesky of Ky once
    
    #draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),[n_mc_smps,M,nx]),perm=[0,2,1])
    return draw_reshape    
     
def get_GP_samples(minibatch, gp_params): ##,med_cov_grid
    """
    returns samples from GP at evenly-spaced gridpoints
    """ 
    #Unravel minibatch object
    Y = minibatch.Y
    T = minibatch.T
    X = minibatch.X 
    ind_kf = minibatch.ind_kf
    ind_kt = minibatch.ind_kt
    num_obs_times = minibatch.num_obs_times
    num_obs_values = minibatch.num_obs_values 
    num_tcn_grid_times = minibatch.num_tcn_grid_times
    #cov_grid = minibatch.cov_grid

    n_mc_smps, M, pad_before = gp_params.n_mc_smps, gp_params.input_dim, gp_params.pad_before
    grid_max = tf.shape(X)[1]
    Z = tf.zeros([0,grid_max, M])
    
    N = tf.shape(T)[0] #number of observations
        
    #setup tf while loop (have to use this bc loop size is variable)
    def cond(i,Z):
        return i<N
    
    def body(i,Z):
        Yi = tf.reshape(tf.slice(Y,[i,0],[1,num_obs_values[i]]),[-1]) #MM: tf.reshape(x, [-1]) flattens tensor x (e.g. [2,3,1] to [6]), slice cuts out all Y data of one patient
        Ti = tf.reshape(tf.slice(T,[i,0],[1,num_obs_times[i]]),[-1])
        ind_kfi = tf.reshape(tf.slice(ind_kf,[i,0],[1,num_obs_values[i]]),[-1])
        ind_kti = tf.reshape(tf.slice(ind_kt,[i,0],[1,num_obs_values[i]]),[-1])
        Xi = tf.reshape(tf.slice(X,[i,0],[1,num_tcn_grid_times[i]]),[-1])
        X_len = num_tcn_grid_times[i]
                
        GP_draws = draw_GP(Yi,Ti,Xi,ind_kfi,ind_kti, gp_params=gp_params)
        pad_len = grid_max-X_len #pad by this much
        #padding direction:
        if pad_before:
            print('Padding GP_draws before observed data..')
            padded_GP_draws = tf.concat([tf.zeros((n_mc_smps,pad_len,M)), GP_draws],1) 
        else:
            padded_GP_draws = tf.concat([GP_draws,tf.zeros((n_mc_smps,pad_len,M))],1) 

        #if lab_vitals_only:
        Z = tf.concat([Z,padded_GP_draws],0) #without covs
        #if covs are used:
        #    medcovs = tf.slice(cov_grid,[i,0,0],[1,-1,-1])
        #    tiled_medcovs = tf.tile(medcovs,[n_mc_smps,1,1])
        #    padded_GPdraws_medcovs = tf.concat([padded_GP_draws,tiled_medcovs],2)
        #    Z = tf.concat([Z,padded_GPdraws_medcovs],0) #with covs
        
        return i+1,Z  
    
    i = tf.constant(0)
    #with tf.control_dependencies([tf.Print(tf.shape(ind_kf), [tf.shape(ind_kf), tf.shape(ind_kt), num_obs_values], 'ind_kf & ind_kt & num_obs_values')]):
    i,Z = tf.while_loop(cond,body,loop_vars=[i,Z],
            shape_invariants=[i.get_shape(),tf.TensorShape([None,None,None])])

    return Z
