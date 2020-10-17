import numpy as np
import sklearn
import sklearn.ensemble
import os
import pickle
import lightgbm as lgb
import preprocess

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

FILE=open('train_gs.dat','r')
train_array=[]
train_gs=[]
for line in FILE:
    line=line.strip()
    table=line.split('\t')
#    whole_train=np.loadtxt(table[0])
    whole_train= load_challenge_data(table[0])
    GS=open(table[1],'r')
    gs=[]
    for gsline in GS:
        gsline=gsline.rstrip()
        gs.append(gsline)
    GS.close()
    pos_start=1000000
    try:
        the_index=gs.index('1')
        pos_start=the_index
    except:
        pass

    #.eg. start: 10, pos_start=4
    i=0
    while ((i<pos_start-6) and (i<whole_train.shape[0])):
        val=(-0.05)/2.0/2.0+0.5;
        train_gs.append(val)
        i=i+1
    while ((i<(pos_start)) and (i<whole_train.shape[0])):
        val=(1/6.0*(6-pos_start+i))/2.0/2.0+0.5
        train_gs.append(val)
        i=i+1
    while ((i<(pos_start+9)) and (i<whole_train.shape[0])):
        val=(1.0-1/9.0*(i-pos_start)+2.0/9.0*(i-pos_start))/2.0/2.0+0.5
        train_gs.append(val)
        i=i+1
    while (i<whole_train.shape[0]):
        if (pos_start==1000000):
            val=(-0.05)/2.0/2.0+0.5
            train_gs.append(val)
        else:
            val=1
            train_gs.append(val)
        i=i+1


    a=whole_train.shape[0]+1
    i=1
    while(i<a):
        data=whole_train[0:i,:]
        processed_data=preprocess.preprocess(data)
        matrix=preprocess.feature(processed_data)
        train_array.append(matrix)
        
        i=i+1

lgb_train = lgb.Dataset(np.asarray(train_array), np.asarray(train_gs))

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 50,
    'learning_rate': 0.05,
    'verbose': 0,
    'n_estimators': 200,
    'reg_alpha': 2.0,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000
                )

print('Saving model...')
# save model to file

filename = 'finalized_model.sav'
pickle.dump(gbm, open(filename, 'wb'))

