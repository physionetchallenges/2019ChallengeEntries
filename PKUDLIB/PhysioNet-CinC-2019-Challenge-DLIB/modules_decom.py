'''
April 2019 by WHX
Modified from
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf
import numpy as np

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def topk_self_attention(inputs, values, num_heads, event_nums, size, scope = 'embed_attention'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(inputs, size, use_bias = False, activation=tf.nn.relu) # (N, T, ev, C)
        K = tf.layers.dense(inputs, size, use_bias = False, activation=tf.nn.relu) # (N, T, ev, C)
        V = tf.layers.dense(values, size, use_bias = False, activation=tf.nn.relu) # (N, T, ev ,C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=3), axis=0) # (h*N, T, ev, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=3), axis=0) # (h*N, T, ev, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=3), axis=0) # (h*N, T, ev, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 1, 3, 2])) # (h*N, T, ev, ev)
        outputs = outputs / (size**0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1))) # (N, T, ev)
        key_masks = tf.tile(key_masks, [num_heads, 1, 1]) # (h*N, T, ev)
        key_masks = tf.tile(tf.expand_dims(key_masks, 2), [1, 1, event_nums, 1]) # (h*N, T, ev, ev)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T, ev, ev)
        
        # Topk remaining 
        '''
        paddings = tf.ones_like(outputs)*(-2**32+1)
        Mask = tf.expand_dims(tf.convert_to_tensor(hp.mask, dtype = tf.bool), 0)
        Mask = tf.tile(Mask, [tf.shape(outputs)[0], 1, 1])
        outputs = tf.where(Mask, paddings, outputs)
        '''
        
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T, ev, ev)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1))) # (N, T, ev)
        query_masks = tf.tile(query_masks, [num_heads, 1, 1]) # (h*N, T, ev)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, 1, event_nums]) # (h*N, T, ev, ev)
        outputs *= query_masks # broadcasting. (N, T, ev, ev)
        
        outputs = tf.matmul(outputs, V_) # (h*n, T, ev, C)
        outputs = tf.concat(tf.split(outputs, num_heads, axis = 0), axis = 3) # (n, T, ev, C)
        return outputs

def pooling(inputs, event_nums, heads, units, 
               l2r, scope = 'multi_pooling'):
    list = []
    n = tf.shape(inputs)[0]
    t = inputs.shape[1]
    for i in range(heads):
        with tf.variable_scope(scope + '_{}'.format(i), reuse = tf.AUTO_REUSE):
            q = tf.layers.dense(inputs, units, use_bias = False, activation = tf.nn.relu)
            v = tf.layers.dense(inputs, units, use_bias = False, activation = tf.nn.relu)
            head_mask = tf.get_variable(name = 'head_mask',
                                        dtype=tf.float32,
                                        shape=[1,1,1,units],
                                        initializer=tf.contrib.layers.xavier_initializer())
            tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(l2r)(head_mask))
            attention = tf.reduce_sum(q * tf.tile(head_mask, (n, t, event_nums,1)), axis = -1)
            attention = tf.expand_dims(tf.nn.softmax(attention), axis = -1)  
            list += [v * attention]
    
    list = tf.concat(list, axis = -1)
    #outputs = tf.nn.max_pool(list, [1,1,event_nums,1], [1,1,1,1], padding = 'VALID', name = 'max_pool')[:,:,0,:]
    outputs = tf.reduce_sum(list, axis = 2)
    return outputs

def att_embedding(inputs, numerical_nums, discret_nums, units, 
           pooling_heads, num_heads, l2r, scope = 'embedding'):
    num_inputs = inputs[:,:,:numerical_nums]
    discret_inputs = tf.to_int32(inputs[:,:,numerical_nums:])
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        w1 = tf.get_variable(name='num_emb_mat',
                             dtype=tf.float32,
                             shape=[numerical_nums, units],
                             initializer=tf.contrib.layers.xavier_initializer())
        wi = tf.get_variable(name='num_emb_index',
                             dtype=tf.float32,
                             shape=[numerical_nums, units],
                             initializer=tf.contrib.layers.xavier_initializer())
        w2 = tf.get_variable(name='diecret_lookup_table',
                             dtype=tf.float32,
                             shape=[2*discret_nums + 1, units],
                             initializer=tf.contrib.layers.xavier_initializer())
        emb_n = tf.tile(tf.expand_dims(num_inputs, -1), [1,1,1,units])
        num_index = tf.ones_like(emb_n) * wi
        emb_n = emb_n * w1 + tf.where(tf.not_equal(emb_n, 0), num_index, tf.zeros_like(emb_n))
        emb_d = tf.nn.embedding_lookup(w2, discret_inputs)
        index = tf.concat((num_index, emb_d), axis = 2)        
        emb = tf.concat((emb_n,emb_d), axis = 2)
        #emb = topk_self_attention(index, emb, num_heads, numerical_nums + discret_nums, units) #(n, T, ev, C)
        #emb = topk_self_attention(tf.concat((num_index, emb_d), axis = 2), emb, pooling_heads, numerical_nums + discret_nums, units)
        emb = pooling(emb, numerical_nums + discret_nums, pooling_heads, units, l2r)
        #emb = tf.nn.max_pool(emb, [1,1,numerical_nums + discret_nums,1], [1,1,1,1], padding = 'VALID', name = 'max_pool')
        #emb = emb[:,:,0,:]
        return emb
    
def positional_encoding(inputs,
                        timelen,
                        num_units,
                        scope="positional_encoding",
                        reuse=None):
    N = tf.shape(inputs)[0]
    position_ind = tf.tile(tf.expand_dims(tf.range(timelen), 0), [N, 1])
    with tf.variable_scope(scope, reuse=reuse):
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(timelen)], np.float32)

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)
                                                                           
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        return outputs

def selfpos_encoding(inputs,
                     E,T,
                     masking=True,
                     scope="positional_encoding"):
    N  = tf.shape(inputs)[0]
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N,1])
        position_enc = tf.get_variable(name = 'timetable',
                                      shape = [T, E],
                                      dtype = tf.float32,
                                      initializer = tf.contrib.layers.xavier_initializer())
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        if masking:
            mask = tf.tile(tf.reduce_sum(tf.abs(inputs), -1, keep_dims = True), [1,1,E])
            outputs = tf.where(tf.equal(mask, 0), mask, outputs)

        return tf.to_float(outputs)

def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, use_bias = False, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, use_bias = False, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, use_bias = False, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            paddings = tf.ones_like(outputs)*(-2**32+1)
            Mask = tf.expand_dims(tf.convert_to_tensor(hp.mask, dtype = tf.bool), 0)
            Mask = tf.tile(Mask, [tf.shape(outputs)[0], 1, 1])
            outputs = tf.where(Mask, paddings, outputs)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)      
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
        return outputs

def feedforward(inputs, 
                num_units=[2048, 512],
                dropout_rate = 0,
                is_training = True,
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
                                
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = normalize(outputs)
    
        return outputs

def DenseInterpolation(inputs, T, M, units):
    W = np.zeros((M, T), dtype = 'float32')
    for j in range(T):
        s = M * j / T
        for i in range(1,M):
            W[i][j] = np.square(1 - abs(s - i) / M)   
    
    N = tf.shape(inputs)[0]
    W = tf.convert_to_tensor(W)
    W = tf.tile(tf.expand_dims(W, 0),[N, 1, 1])
    outputs = tf.matmul(W, inputs)
    outputs = tf.reshape(outputs, (-1, M * units))
    return outputs
    