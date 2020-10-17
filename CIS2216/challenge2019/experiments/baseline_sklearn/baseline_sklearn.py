from .util import compute_prediction_utility, sample_model
import pickle as pkl
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
import time
from scipy.interpolate import Akima1DInterpolator


def get_global_impute_value():
    arr = [84.58144,97.19395,36.97723,123.75047,82.40010,63.83056,18.72650,32.95766,-0.68992,24.07548,0.55484,7.37893,41.02187,92.65419,260.22338,23.91545,102.48366,7.55753,105.82791,1.51070,1.83618,136.93228,2.64667,2.05145,3.54424,4.13553,2.11406,8.29010,30.79409,10.43083,41.23119,11.44641,287.38571,196.01391,62.00947,0.55927,0.49657,0.50343,-56.12512,26.99499,0.01798]
    return arr

def extract_feats(mat):
    """
    extract features from data, be aware that shouldn't use future data

    mat shape, (n_time, n_col)
    """
    feats = []
    feats.extend(list(np.max(mat[-24:,:], axis=0)))
    feats.extend(list(np.min(mat[-24:,:], axis=0)))
    return feats

def my_impute_train(mat, extract_feats=False):
    ## impute
    global_impute_value = get_global_impute_value()
    n_row, n_col = mat.shape
    for i_col in range(n_col):
        y_raw = mat[:,i_col]
        x_raw = np.arange(len(y_raw))
        valid_idx = ~np.isnan(y_raw)
        x = x_raw[valid_idx]
        y = y_raw[valid_idx]
        try:
            cs = Akima1DInterpolator(x, y)
            y_all = cs(x_raw)
        except:
            y_all = y_raw
        y_all[np.isnan(y_all)] = global_impute_value[i_col]
        mat[:, i_col] = y_all
    
    if extract_feats:
        ## extract feats
        feat = []
        for i_row in range(n_row):
            feat.append(extract_feats(mat[:i_row+1]))
        feat = np.array(feat)
        ## concate
        out_mat = np.concatenate([mat, feat], axis=1)
    else:
        out_mat = mat

    return out_mat

def my_impute_test(mat, extract_feats=False):
    """
    be aware that impute shouldn't use future data
    """
    ## impute
    global_impute_value = get_global_impute_value()
    n_row, n_col = mat.shape
    for i_col in range(n_col):
        prev_value = np.nan
        for i_row in range(n_row):
            if np.isnan(mat[i_row, i_col]):
                mat[i_row, i_col] = prev_value
            else:
                prev_value = mat[i_row, i_col]
        mat[np.isnan(mat[:,i_col]), i_col] = global_impute_value[i_col]

    if extract_feats:
        ## extract feats
        feat = []
        for i_row in range(n_row):
            feat.append(extract_feats(mat[:i_row+1]))
        feat = np.array(feat)
        ## concate
        out_mat = np.concatenate([mat, feat], axis=1)
    else:
        out_mat = mat

    return out_mat

def load_data(seed):
    """
    train 0.9, val 0.05, test 0.05
    """
    ## read and process data
    with open('data/data.pkl', 'rb') as fin:
        res = pkl.load(fin)
    data = res['data']
    label = res['final_label']
    print('entire data stat', Counter(label))
    
    ## train, val, test split
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size = 0.05, random_state=seed)
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size = 0.05, random_state=seed)
    print('train data stat', Counter(label_train), 'val data stat', Counter(label_val), 'test data stat', Counter(label_test))
    
    return data_train, data_val, data_test

def train(seed, method='LR', shift=-1):

    ## load data
    data_train, data_val, data_test = load_data(seed=seed)

    ## make train data
    X_train = []
    Y_train = []
    for data in data_train:

        ## process data
        tmp_data = data.values[:, :-1]
        tmp_data = my_impute_train(tmp_data)
        X_train.append(tmp_data)
        
        ## process label
        tmp_label = data.values[:, -1]
        if shift != -1:
            ## shift label to 6 hours later
            if tmp_label[-1] == 1:
                if len(tmp_label) < shift:
                    tmp_label = np.array([1]*len(tmp_label))
                else:
                    tmp_label = np.append(tmp_label[shift:], [1] *shift)
        else:
            ## shift label to the last label
            if tmp_label[-1] == 1:
                tmp_label = np.array([1]*len(tmp_label))
            else:
                tmp_label = np.array([0]*len(tmp_label))

        Y_train.append(tmp_label)
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    print('flatten train data shape', X_train.shape, Y_train.shape)    
    
    ## make val data
    X_val = []
    Y_val = []
    for data in data_val:
        ## process data
        tmp_data = data.values[:, :-1]
        tmp_data = my_impute_test(tmp_data)
        X_val.append(tmp_data)
        Y_val.append(data.values[:, -1])

    ## make test data
    X_test = []
    Y_test = []
    for data in data_test:
        ## process data
        tmp_data = data.values[:, :-1]
        tmp_data = my_impute_test(tmp_data)
        X_test.append(tmp_data)
        Y_test.append(data.values[:, -1])

    ## trivial baseline
    score_sum = [0, 0, 0, 0]
    for i in range(len(X_test)):
        pred_1 = [0]*len(Y_test[i])
        tmp_score, _ = compute_prediction_utility(Y_test[i], pred_1)
        score_sum[0] += tmp_score
        pred_2 = [1]*len(Y_test[i])
        tmp_score, _ = compute_prediction_utility(Y_test[i], pred_2)
        score_sum[1] += tmp_score
        pred_3 = np.random.randint(2, size=len(Y_test[i]))
        tmp_score, _ = compute_prediction_utility(Y_test[i], pred_3)
        score_sum[2] += tmp_score
        pred_4 = sample_model(X_test[i])
        tmp_score, _ = compute_prediction_utility(Y_test[i], pred_4)
        score_sum[3] += tmp_score
    baseline_scores = np.array(score_sum)/len(X_test)
    print('All 0: {0:.4f}, All 1: {1:.4f}, Random: {2:.4f}, Sample: {3:.4f}'.format(baseline_scores[0], baseline_scores[1], baseline_scores[2], baseline_scores[3]))
        
    ## build model
    if method == 'LR':
        clf = LR()
    elif method == 'RF':
        clf = RF(n_estimators=100)

    ### train
    clf.fit(X_train, Y_train)
    
    ### val to get best thresh
    all_score = []
    all_thresh = np.arange(0.0,0.5,0.01)
    for thresh in all_thresh:
        score_sum = 0
        for i in range(len(X_val)):
            tmp_val = X_val[i]
            tmp_pred_proba = clf.predict_proba(tmp_val)[:, 1]
            tmp_pred = np.array(tmp_pred_proba > thresh, dtype=np.int32)
            tmp_score, _ = compute_prediction_utility(Y_val[i], tmp_pred)
            score_sum += tmp_score

        all_score.append(score_sum/len(X_val))
        print(thresh, score_sum/len(X_val))
    
    best_thresh = all_thresh[np.argmax(all_score)]
    
    ### test
    score_list = []
    all_pred = []
    for i in range(len(X_test)):
        tmp_test = X_test[i]
        tmp_pred_proba = clf.predict_proba(tmp_test)[:, 1]
        tmp_pred = np.array(tmp_pred_proba > best_thresh, dtype=np.int32)
        tmp_score, _ = compute_prediction_utility(Y_test[i], tmp_pred)
        score_list.append(tmp_score)
        all_pred.append(tmp_pred)

    print('best_thresh', best_thresh, 'score', np.mean(score_list))
    
    return clf, best_thresh, np.mean(score_list), baseline_scores

def get_sepsis_score(mat, model_file):
    mat = mat[:, :40]
    mat = my_impute_test(mat)
    mat = mat[[-1], :]
    with open(model_file, 'rb') as fin:
        res = pkl.load(fin)
    all_pred = []
    for _, v in res.items():
        clf, thresh, _ = v
        tmp_pred_proba = clf.predict(mat)
        tmp_pred = np.array(tmp_pred_proba > thresh, dtype=np.int32)
        all_pred.append(tmp_pred)
    all_pred = np.array(all_pred)
    vote_pred_prob = np.mean(all_pred, axis=0)
    vote_pred = np.array(vote_pred_prob > 0.5, dtype=np.int32)
    
    return vote_pred_prob[0], vote_pred[0]

if __name__ == "__main__":
    
    # ## run for baselines
    # log_name = 'log/log_sklearn.txt'
    # with open(log_name, 'w') as fout:
    #     print('seed,LR,RF,All0,All1,Random,Sample', file=fout)
    # for seed in range(10):
    #     _, _, score_LR, baseline_scores_LR = train(method='LR', seed=seed)
    #     _, _, score_RF, baseline_scores_RF = train(method='RF', seed=seed)
    #     with open(log_name, 'a') as fout:
    #         print('{0},{1},{2},{3},{4},{5},{6}'.format(seed, score_LR, score_RF, baseline_scores_LR[0], baseline_scores_LR[1], baseline_scores_LR[2], baseline_scores_LR[3]), file=fout)
    
    ### run for pars
    # run_id = 'new_feats'
    # log_name = 'log/{0}.txt'.format(run_id)
    # model_file = 'models/{0}.pkl'.format(run_id)
    # res = {}
    # with open(log_name, 'w') as fout:
    #     print('seed,shift,LR,All0,All1,Random,Sample', file=fout)
    # for seed in range(10):
    #     for shift in [-1]:cd ../c 
    #         clf, thresh, score, baseline_scores = train(method='LR', shift=shift, seed=seed)
    #         res[seed] = [clf, thresh, score]
    #         with open(model_file, 'wb') as fout:
    #             pkl.dump(res, fout)
    #         with open(log_name, 'a') as fout:
    #             print('{0},{1},{2},{3},{4},{5},{6}'.format(seed, shift, score, baseline_scores[0], baseline_scores[1], baseline_scores[2], baseline_scores[3]), file=fout)

    with open('../data/data_small.pkl', 'rb') as fin:
        res = pkl.load(fin)
    data = res['data']
    for d in data:
        d = d.values
        print(get_sepsis_score(d, '../models/lr_for_submit.pkl'))



