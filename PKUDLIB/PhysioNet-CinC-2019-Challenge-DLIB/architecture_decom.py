'''
April 2019 by WHX
Modified from
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from modules_decom import * 
from hyperparams import Hyperparams_decom as hp
def bnl(logits, labels):
    l = tf.to_float(labels)
    return -(l * tf.log(tf.clip_by_value(logits, 1e-8, 1)) + 
          (1-l) * tf.log(tf.clip_by_value(1 - logits, 1e-8, 1)))   
class Graph():
    def __init__(self, mode):
        self.x = tf.placeholder(tf.float32, shape=(None, hp.timelen, hp.event_nums))
        self.y = tf.placeholder(tf.int32, shape=(None))
        self.T = tf.placeholder(tf.bool)
        # Encoder
        with tf.variable_scope("encoder"):
            '''
            ## Embedding
            self.enc = tf.layers.dense(self.x, hp.hidden_units, use_bias = False, name='embedding')
            '''
            ## he_embedding
            self.enc = att_embedding(self.x, 37, 3, hp.hidden_units, hp.pool_heads, hp.num_heads, hp.l2r)
            
            units = hp.hidden_units * hp.pool_heads
            ## Positional Encoding
            self.enc += selfpos_encoding(self.x,
                                         units,
                                         hp.timelen, 
                                         scope="enc_pe")
            
            # Dropouts
            self.enc = tf.layers.dropout(self.enc, rate = hp.dropout_rate, training=self.T)
                    
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc, 
                                                   keys=self.enc, 
                                                   num_units = units,
                                                   num_heads=hp.num_heads, 
                                                   dropout_rate=hp.dropout_rate,
                                                   is_training = self.T,
                                                   causality=False)
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4 * units, units], is_training = self.T)
                           
            ## Liner projection
            self.enc = tf.reshape(self.enc, (-1, hp.timelen * units))
            self.enc = tf.layers.dense(self.enc, units = units, activation = tf.nn.relu, name = 'lp')  
            #self.enc = DenseInterpolation(self.enc, hp.timelen, hp.DI_M, hp.hidden_units)
            #self.enc = tf.layers.max_pooling1d(self.enc, hp.timelen, 1, padding = 'valid')[:,0,:] 
            print(self.enc.shape) 
            
        # Final linear projection
        logits = tf.layers.dense(self.enc, units = 1, activation = tf.nn.sigmoid)[:,0]
        ones = tf.ones_like(logits)
        cost = tf.where(tf.equal(self.y, 1), ones, ones*hp.class_weighted)
        self.preds = logits
        #self.mean_loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.one_hot(self.y,2))
        self.lloss = bnl(logits = logits, labels = self.y)
        mean_loss = tf.reduce_sum(cost * self.lloss) / tf.reduce_sum(cost)
        tf.add_to_collection('losses', mean_loss)
        self.mean_loss = tf.add_n(tf.get_collection('losses'))
        self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
        self.learning_rate = tf.Variable(hp.learning_rate, name = 'lr', trainable = False)
        #al = tf.train.exponential_decay(hp.learning_rate, self.global_step, 4251, 0.9, staircase = True)
        #self.optimizer = tf.train.GradientDescentOptimizer(al)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.mean_loss)
