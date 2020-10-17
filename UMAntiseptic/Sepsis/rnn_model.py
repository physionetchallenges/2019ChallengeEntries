import numpy as np
import os
import time as tm

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dropout, BatchNormalization, Flatten, Reshape, Add, Concatenate, LSTM, Masking, GRU
from keras.models import Model, load_model
from keras.optimizers import Adadelta, Adam
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.metrics import binary_accuracy
import h5py

import my_evaluate_sepsis_score

batch_size = 300
mask_value = -9999.0

data_folder = '../../Data/Sets/'
train_dir = data_folder+'Train/'
val_dir = data_folder+'Validation/'
test_dir = data_folder+'Test/'

def read_challenge_data(input_file):
	with open(input_file, 'r') as f:
		header = f.readline().strip()
		column_names = header.split('|')
		data = np.loadtxt(f, delimiter='|')
	return column_names, data


def read_data():
	train_files = []
	for filename in os.listdir(train_dir):
		full_filename = os.path.join(train_dir, filename)
		if os.path.isfile(full_filename) and full_filename.endswith('.psv'):
			train_files.append(filename)
	
	train_files = sorted(train_files)
	
	val_files = []
	for filename in os.listdir(val_dir):
		full_filename = os.path.join(val_dir, filename)
		if os.path.isfile(full_filename) and full_filename.endswith('.psv'):
			val_files.append(filename)
	
	val_files = sorted(val_files)
	
	test_files = []
	for filename in os.listdir(test_dir):
		full_filename = os.path.join(test_dir, filename)
		if os.path.isfile(full_filename) and full_filename.endswith('.psv'):
			test_files.append(filename)
	
	test_files = sorted(test_files)	
	
	train_data = [];
	train_subj_labels = []
	for k in range(len(train_files)):
		column_names, data = read_challenge_data(train_dir+train_files[k])
		train_subj_labels.append(any(data[:,-1]))
		train_data.append(data)
	
	val_data = [];
	val_subj_labels = []
	for k in range(len(val_files)):
		column_names, data = read_challenge_data(val_dir+val_files[k])
		val_subj_labels.append(any(data[:,-1]))
		val_data.append(data)
	
	test_data = [];
	test_subj_labels = []
	for k in range(len(test_files)):
		column_names, data = read_challenge_data(test_dir+test_files[k])
		test_subj_labels.append(any(data[:,-1]))
		test_data.append(data)	
	
	return train_data, train_subj_labels, val_data, val_subj_labels, test_data, test_subj_labels, column_names

step_n = 10

def eval_model(data, model, thresh=list(range(0,step_n))):
	y_true = []
	scores = []
	for si in range(len(data)):
		subj = data[si]
		
		subj[np.isnan(subj)] = 0
				
		slen = subj.shape[0]
		
		batch_n = int(np.ceil(subj.shape[0]/batch_size))
		subj = np.concatenate([subj,np.full([batch_n*batch_size-subj.shape[0],subj.shape[1]],mask_value)])			
		
		subj = np.reshape(subj,(subj.shape[0],1,subj.shape[1]))		
		
		y_true.append(subj[:slen,0,-1].tolist())
		y_hat = model.predict(subj[:,:,:-1], batch_size=batch_size, verbose=0)
		y_hat = [y_hat[i][0] for i in range(slen)]
		scores.append(y_hat)
		model.reset_states()
	
	accuracies = []
	f_measures = []
	utils = []
	for th in thresh:
		y_pred = [[int(scores[i][j]>th/step_n) for j in range(len(scores[i]))] for i in range(len(scores))]
		auroc, auprc, accuracy, f_measure, normalized_observed_utility = my_evaluate_sepsis_score.evaluate_scores(y_true, y_pred, scores)
		accuracies.append(accuracy)
		f_measures.append(f_measure)
		utils.append(normalized_observed_utility)
	
	idx = np.argmax(utils)
	
	return auroc, auprc, accuracies[idx], f_measures[idx], utils[idx], thresh[idx]

	
def train_model():

train_data, train_subj_labels, val_data, val_subj_labels, test_data, test_subj_labels, column_names = read_data()
train_data_small = train_data[:2000]

p_idx = [i for i in range(len(train_data)) if train_subj_labels[i]==1]
n_idx = [i for i in range(len(train_data)) if train_subj_labels[i]==0]

#lns = [len(train_data[i]) for i in range(len(train_data))]
#batch_size = np.min(lns)

input_dim = len(column_names)-1
#input_dim = 1

input = Input(batch_shape=(batch_size, None,input_dim))
#x = Dense(15, activation=None)(input)
x = Masking(mask_value=mask_value)(input)
x = GRU(32,stateful=True)(x)
x = GRU(32,stateful=True)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=output)

model.summary()

model.compile(optimizer='rmsprop', loss='mean_squared_error')

epoch_n = 100

for ep in range(epoch_n):
	p_idx = np.random.permutation(p_idx)
	n_idx = np.random.permutation(n_idx)
	
	tm1 = tm.time()
	pi = 0
	for ni in range(len(n_idx)):
		n_subj = train_data[n_idx[ni]]			
		p_subj = train_data[p_idx[pi]]			
		
		n_subj[np.isnan(n_subj)] = 0
		p_subj[np.isnan(p_subj)] = 0
		
		idx = np.argmax(p_subj[:,-1])
		p_subj[np.max([0,idx-6]):,-1] = 1		
		
		#n_subj  = np.concatenate([n_subj[:,-1:],n_subj],axis=1)
		#p_subj  = np.concatenate([p_subj[:,-1:],p_subj],axis=1)		
		
		batch_n = int(np.ceil(n_subj.shape[0]/batch_size))
		n_subj = np.concatenate([n_subj,np.full([batch_n*batch_size-n_subj.shape[0],n_subj.shape[1]],mask_value)])
		
		batch_n = int(np.ceil(p_subj.shape[0]/batch_size))
		p_subj = np.concatenate([p_subj,np.full([batch_n*batch_size-p_subj.shape[0],p_subj.shape[1]],mask_value)])				
		
		n_subj = np.reshape(n_subj,(n_subj.shape[0],1,n_subj.shape[1]))
		p_subj = np.reshape(p_subj,(p_subj.shape[0],1,p_subj.shape[1]))		
		
		pi += 1
		if(pi>=len(p_idx)):
			pi = 0
			p_idx = np.random.permutation(p_idx)
			print('Iter: %i - Elapsed Time: %.3f' % (ni, tm.time()-tm1))
		
		hist = model.fit(p_subj[:,:,:-1], p_subj[:,:,-1], epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
		
		hist = model.fit(n_subj[:,:,:-1], n_subj[:,:,-1], epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	
	auroc, auprc, accuracies, f_measures, utils, thresh = eval_model(val_data, model)
	
	print('Epoch: %i - AUROC: %.3f - AUPRC: %.3f - Acc: %.3f - F1: %.3f - Util: %.3f' % (ep, auroc, auprc, accuracies, f_measures, utils))

auroc, auprc, accuracies, f_measures, utils, thresh = eval_model(test_data, model, [thresh])

print('Test results: AUROC: %.3f - AUPRC: %.3f - Acc: %.3f - F1: %.3f - Util: %.3f' % (auroc, auprc, accuracies, f_measures, utils))

model.save('rnn_model.h5')