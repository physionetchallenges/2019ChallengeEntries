import numpy as np
import tensorflow as tf
flags = tf.app.flags.FLAGS


class Model:
	def build_model(self, inputs, dropout, augmentation, istraining):
		self.dropout_conv = tf.gather(dropout, 0)
		self.dropout_lstm1 = tf.gather(dropout, 1)
		self.dropout_lstm2 = tf.gather(dropout, 2)
		self.dropout_lstm3 = tf.gather(dropout, 3)
		self.dropout_fc = tf.gather(dropout, 4)

		self.inputs = tf.identity(inputs[0])
		self.inputs2 = tf.identity(inputs[1])
		self.augmentation = augmentation
		self.istraining = istraining

		with tf.variable_scope(flags.name, reuse=tf.AUTO_REUSE):
			self.encoded = self.encoder([self.inputs, self.inputs2])
			self.output = self.classifier(self.encoded, flags.output_size)
			self.output2 = self.classifier(self.encoded, 1, 'classifier_aux')

	def encoder(self, inputs, name='encoder'):
		self.dropout = [self.dropout_lstm1, self.dropout_lstm2, self.dropout_lstm3]
		istraining = self.istraining
		
		### Local perturbation
		M, inputs = inputs
		delta = inputs[:,:,0:34] 
		mask = inputs[:,:,34:68] 
		#ddata = inputs[:,:,68:102] 
		data = inputs[:,:,102:136]
		demo = inputs[:,:,-6:]
		
		data = self.input_imputation(data, mask, delta, istraining)
		ddata = tf.concat([tf.zeros_like(data[:,0:1,:]), data[:,1:,:,]-data[:,:-1,:]], 1)
		ddata = tf.stop_gradient(ddata)
		inputs = tf.concat([mask, delta, ddata, data, demo], -1)

		## Positional encoding
		encoded_pos = self.positional_encoder(inputs, 512, 'inputs_pos')
		
		## Encoder 
		encoded = encoded_pos
		for i in range(flags.model_depth):
			encoded = self.transformer_block(encoded, M, 'en'+str(i))
		outputs = encoded
		decoded = encoded
		
		## Decoder 
		self.enc_features = tf.layers.conv1d(decoded, 68, 1, 1, 'valid',
									kernel_initializer=tf.contrib.layers.xavier_initializer())
		return outputs
	
	def input_imputation(self, data, mask, delta, istraining, name='input_imputation'):
		with tf.variable_scope(name):
			delta = tf.layers.dense(delta, delta.get_shape()[-1]*2, activation=None)
			gamma = tf.exp(-tf.maximum(0.0, delta))
			gamma_m, gamma_s = tf.split(gamma, 2, -1)
			dist = tf.distributions.Normal(loc=0.0, scale=1-gamma_s)
			noise = tf.squeeze(dist.sample([1]))	# noise:[0,1]
			noise = tf.cond(istraining, lambda:noise, lambda:tf.zeros_like(noise))
			data = mask*data + (1-mask)*gamma_m*data + noise
			return data
	
	def positional_encoder(self, inputs, num_units, name):
		with tf.variable_scope(name):
			units = int(inputs.get_shape()[-1])
			PE = self.positional_encoding(inputs, num_units=units, scope='enc_pe')
			encoded = inputs + PE	# (batch, time, feature)
			encoded = tf.layers.conv1d(encoded, num_units, 1, 1, 'valid',
																kernel_initializer=tf.contrib.layers.xavier_initializer())
			return encoded

	def positional_encoding(self, inputs,
																num_units,
																zero_pad=False,
																scale=False,
																scope="positional_encoding",
																reuse=None):
		N,T,D = inputs.get_shape().as_list()
		with tf.variable_scope(scope, reuse=reuse):
			position_ind = tf.tile(tf.expand_dims(tf.range(T),0), [N, 1])
			# First poart of PE: sin and cos argument
			position_enc = np.array([[pos/np.power(10000, 2.*i/num_units) 
																	for i in range(num_units)] for pos in range(T)], dtype='f')
			# Second part, apply the cosine to even columns and sin to odds
			position_enc[:,0::2] = np.sin(position_enc[:,0::2])
			position_enc[:,1::2] = np.cos(position_enc[:,1::2])

			# Convert to a tensor
			lookup_table = tf.convert_to_tensor(position_enc)
			outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

			if scale:
				outputs = outputs*num_units**0.5

			return outputs

	def transformer_block(self, encoded, M, name):
		with tf.variable_scope(name):
			# Multihead attention
			encoded += self.multihead_attention(queries=encoded,
																			keys=encoded,
																			mask=M,
																			num_units=512,
																			num_heads=8,
																			multihead_axis=-1,
																			keep_prob=1-self.dropout[0],
																			is_training=self.istraining,
																			causality=False,
																			name='block')
			encoded = tf.contrib.layers.layer_norm(encoded)

			# Feedfowrd
			encoded += self.FFN(inputs=encoded,
													kernel_size=3,
													activation=self.gelu,
													scope='FFN')
			encoded = tf.contrib.layers.layer_norm(encoded)
			return encoded
	
	def multihead_attention(self, queries,
																keys,
																mask,
																num_units=None,
																num_heads=8, 
																multihead_axis=2, #1:time, 2:features
																keep_prob=1.0, 
																is_training=True, 
																causality=True,
																name="multihead_attention", 
																reuse=None):
			
		with tf.variable_scope(name,reuse=reuse):
			if num_units is None:
					num_units = queries.get_shape().as_list[-1]

			# Linear projection
			Q = tf.layers.dense(queries, num_units, activation=None)	# (N, T_q, C)
			K = tf.layers.dense(keys, num_units, activation=None)	# (N, T_k, C)
			V = tf.layers.dense(keys, num_units, activation=None)	# (N, T_k, C)
			# Split and concat
			Q_ = tf.concat(tf.split(Q, num_heads, axis=multihead_axis), axis=0)	# (h*N, T_q, C/h)
			K_ = tf.concat(tf.split(K, num_heads, axis=multihead_axis), axis=0)	# (h*N, T_k, C/h)
			V_ = tf.concat(tf.split(V, num_heads, axis=multihead_axis), axis=0)	# (h*N, T_k, C/h)
			# Multiplication
			attn = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))	# (h*N, T_q, T_k)
			attn = attn / (K_.get_shape().as_list()[-1]**0.5)
			# Masking
			attn = self.mask(attn, mask, type='predefined')
			if causality:
				attn = self.mask(attn, type='future')
			# Activation
			attn = tf.nn.softmax(attn) # axis:T_k
			attn = tf.nn.dropout(attn, keep_prob=keep_prob)	# (h*N, T_q, T_k) 
			outputs = tf.matmul(attn, V_)	
			# Restore shape
			outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=multihead_axis) 

			return outputs
	
	def FFN(self, inputs,
								kernel_size,
								activation,
								scope="feedforward_network",
								reuse=None):
			
		with tf.variable_scope(scope, reuse=reuse):
			encoded = tf.layers.separable_conv1d(inputs, 512, kernel_size, 1, 'same',
																depthwise_initializer=tf.contrib.layers.xavier_initializer(),
																pointwise_initializer=tf.contrib.layers.xavier_initializer())
			encoded = activation(encoded)
			encoded = tf.layers.separable_conv1d(encoded, 512, kernel_size, 1, 'same',
																depthwise_initializer=tf.contrib.layers.xavier_initializer(),
																pointwise_initializer=tf.contrib.layers.xavier_initializer())
			return encoded
	
	def mask(self, inputs, mask=None, queries=None, keys=None, type=None):
		padding_num = -2 ** 23 + 1
		if type in ('predefined'):
			h = int(int(inputs.shape[0])/int(mask.shape[0]))
			masks = tf.concat([mask]*h, 0)
			masks = tf.expand_dims(masks, -1)
			masks = tf.concat([masks]*int(inputs.shape[-1]) ,-1)
			paddings = tf.ones_like(inputs) * padding_num
			outputs = tf.where(tf.equal(masks, 0), paddings, inputs)

		elif type in ('K', 'key', 'keys'):
			masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) # (N, T_k)
			masks = tf.expand_dims(masks, 1)
			masks = tf.tile(masks, [1, tf.shape(queries)[1], 1]) # (N, T_q, T_k)
			# Apply masks to inputs
			paddings = tf.ones_like(inputs) * padding_num
			outputs = tf.where(tf.equal(masks, 0), paddings, inputs) # (N, T_q, T_k)
		
		elif type in ('Q', 'query', 'queries'):
			masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
			masks = tf.expand_dims(masks, -1)
			masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])
			# Apply masks to inputs
			outputs = inputs*mask
		
		elif type in ('f', 'future', 'right'):
			diag_vals = tf.ones_like(inputs[0,:,:])	# (T_q, T_l)
			tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
			masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1]) # (N, T_q, T_k)
			paddings = tf.ones_like(masks)*padding_num
			outputs = tf.where(tf.equal(mask, 0), paddings, inputs)
	
		return outputs

	def gelu(self, x):
		return 0.5 * x * (1 + tf.nn.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

	def classifier(self, inputs, output_size, name='classifier'): 
		drop = self.dropout_fc
		istraining = self.istraining
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			# Linear classifier
			#print('Classifier')
			#inputs = tf.reshape(inputs, [-1, inputs.get_shape()[-1]])
			#output = tf.layers.flatten(inputs)
			output = tf.layers.dropout(inputs, drop, training=istraining)
			#output = tf.layers.batch_normalization(output, training=istraining)
			output = tf.layers.dense(output, output_size, activation=None)
			#output = tf.reshape(output, [-1, 336, output_size])
			return output










