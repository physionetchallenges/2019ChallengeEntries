import os
import tensorflow as tf
import numpy as np
from model import *
flags = tf.app.flags.FLAGS

class Build(Model):
	def __init__(self, config):
		self.model = Model.build_model
		self.build_graph()
	
	def build_graph(self):
		N = flags.batch_size
		T = flags.input_length 
		
		self.X = tf.placeholder(tf.float32, [N, T, 34*4+6])
		#self.Y = tf.placeholder(tf.int32, [N, T])
		#self.Z = tf.placeholder(tf.float32, [N, T, 34*4+6])
		self.M = tf.placeholder(tf.float32, [N, T])
		self.DROPOUT = tf.placeholder(tf.float32, 5)
		#self.current_iteration = tf.placeholder(tf.float32)
		self.AUGMENTATION = tf.placeholder(tf.float32, 3)
		self.istraining = tf.placeholder(tf.bool)
		
		self.model(self, [self.M, self.X], self.DROPOUT, self.AUGMENTATION, self.istraining)
		self.probs = tf.nn.softmax(self.output[:,-1,:])
		
	

