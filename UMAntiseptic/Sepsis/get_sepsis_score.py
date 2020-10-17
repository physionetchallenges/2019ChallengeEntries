#!/usr/bin/env python

import sys
import numpy as np
import os, shutil, zipfile
import h5py

from keras.models import load_model

batch_size = 300
mask_value = -9999.0
thresh = 7/10

def get_sepsis_score(subj, model):
	subj[np.isnan(subj)] = 0
	
	slen = subj.shape[0]
	
	batch_n = int(np.ceil(subj.shape[0]/batch_size))
	subj = np.concatenate([subj,np.full([batch_n*batch_size-subj.shape[0],subj.shape[1]],mask_value)])			
	
	subj = np.reshape(subj,(subj.shape[0],1,subj.shape[1]))		
	
	model.reset_states()
	scores = model.predict(subj, batch_size=batch_size, verbose=0)
	scores = np.array([scores[i][0] for i in range(slen)])
	model.reset_states()	
	
	labels = (scores > thresh)
	
	score = scores[-1]
	label = labels[-1]
	return score, label

def load_sepsis_model():
	model = load_model('rnn_model.h5')
	return model
