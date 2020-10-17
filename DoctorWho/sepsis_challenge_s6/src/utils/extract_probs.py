import numpy as np

def extract_probs(padded_probs, mask):
    """
    Inputs: 
    -padded probs of a minibatch [n_samples, max_batch_len] 
    -mask indicating valid points of the probs array of the minibatch [n_samples, max_batch_len]
    Returns:
    -list of arrays containing time-point-wise probabilities per patient   
    """
    probs = []    
    n_samples = padded_probs.shape[0] 
    for i in np.arange(n_samples):
        sample_probs = padded_probs[i, mask[i]]                 
    pass


