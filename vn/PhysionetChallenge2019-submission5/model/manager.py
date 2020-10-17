import os
import tensorflow as tf
import numpy as np
from utils import *
from preprocessing import *

flags = tf.app.flags.FLAGS


class Manager:
	def __init__(self, config, model):
		#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
		#os.environ["CUDA_VISIBLE_DEVICES"]='4'
				
		self.epoch = 0
		self.print_user_flags = config.print_user_flags
		# get_result
		self.graph = tf.get_default_graph()
		self.X = model.X
		self.M = model.M
		self.DROPOUT = model.DROPOUT
		self.AUGMENTATION = model.AUGMENTATION
		self.istraining = model.istraining
		self.probs = model.probs
		self.session_launcher()

	def variable_initializer(self, sess, DIR_SOURCE):
		load_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
															scope=flags.LOAD_SCOPE)
		self.saver = tf.train.Saver(load_vars, max_to_keep=3)
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(DIR_SOURCE)
		self.saver.restore(sess, ckpt.model_checkpoint_path)

	def iter_fn(self, sess, data):
		prob = sess.run(self.probs,
										feed_dict={
											self.X:data[:,:,:-1],
											self.M:data[:,:,-1],
											self.DROPOUT:[0.0,0.0,0.0,0.0,0.0], 
											self.AUGMENTATION:[0,0,0],
											self.istraining:False})	
		return prob[0][1]
	
	def inference(self, data):
		columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',\
				'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',\
				'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',\
				'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',\
				'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',\
				'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
				'HospAdmTime', 'ICULOS'] 
		data = Preprocessing(data, columns)	# 50,C
		data = np.expand_dims(data,0)	# 1,50,C
		score = self.iter_fn(self.sess, data)
		return score

	def session_launcher(self):
		config = tf.ConfigProto(allow_soft_placement=True, 
														log_device_placement=False)
		config.gpu_options.allow_growth=True
		self.sess = tf.Session(config=config)
		self.variable_initializer(self.sess, flags.DIR_SOURCE)


