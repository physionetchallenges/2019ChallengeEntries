import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from compute_scores_2019 import compute_accuracy_f_measure, compute_prediction_utility, compute_auc, compute_scores_2019
from explore_data import read_challenge_data, get_columns_names
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
# plot feature importance manually
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from matplotlib import pyplot
from util import *
import xgboost as xgb
import glob
import pickle


def prepare_data(file_name, record_num, train_num, test_num):
    print('Loading data...')
    # read the data
    # processed_df = pd.read_csv('D:/Projects/physionet-master/training_data/imputed_scaled_stage1_data.csv')
    processed_df = pd.read_csv(file_name)
    column_names = processed_df.columns.values
    X, y = processed_df[column_names[:-1]].values, processed_df[column_names[-1]].values

    print('Splitting data...')
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_num, test_size=test_num, shuffle=False)
    valid_num = int(0.2 * train_num)
    X_valid, y_valid = X_train[train_num-valid_num:], y_train[train_num-valid_num:]
    X_train, y_train = X_train[:train_num-valid_num], y_train[:train_num-valid_num]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def tune_xgbt_model(X_train, X_valid, y_train, y_valid):
    # sklearn API
    # init

    print('Tuning by sklearn API...')
    xgbm = xgb.XGBClassifier(
                            tree_method='gpu_hist',
                            objective='binary:logistic',
                            booster='gbtree',
                            n_estimators=300,
                            learning_rate=0.02,
                            max_depth=6,
                            scale_pos_weight=50,

                            gamma=0,
                            min_child_weight=10,
                            max_delta_step=0,
                            subsample=0.6,
                            colsample_bytree=0.8,
                            colsample_bylevel=0.8,
                            colsample_bynode=1,
                            reg_alpha=0.1,
                            reg_lambda=0.1,

                            base_score=0.5,
                            random_state=0,
                            verbosity=1,
                            silent=False,
                            nthread=4,
                            seed=None,
                            missing=None)

    # tune parameters
    param_grid1 = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.02, 0.05, 0.1]
    }

    param_grid2 = {
        'max_depth': [6, 7, 8],
        'scale_pos_weight': [50, 100, 150]
    }
     
    param_grid3 = {
        # 'max_delta_step': [10, 50],
        'min_child_weight': [1, 10, 100]
    }
     
    param_grid4 = {
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    param_grid5 = {
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0, 0.01, 0.1]
    }

    optimal_param = {}
    for para_grid in [param_grid1, param_grid2, param_grid3, param_grid4, param_grid5]:
        para_grid.update(optimal_param)

        # score by balanced_accuracy
        gsearch = GridSearchCV(estimator=xgbm, param_grid=para_grid, scoring='balanced_accuracy', cv=5, verbose=1, n_jobs=4)   # roc_auc, balanced_accuracy, f1
        gsearch.fit(X_train, y_train)
        print('Best parameters for balanced_accuracy:', gsearch.best_params_)
        print('Best score for balanced_accuracy:', gsearch.best_score_)

        best_params = gsearch.best_params_
        for k in best_params:
            best_params[k] = [best_params[k]]      # put int in a list

        # update optimal param
        optimal_param.update(best_params)


    '''
    # score by recall
    gsearch = GridSearchCV(estimator=xgbm, param_grid=param_grid, scoring='recall', cv=5, verbose=1, n_jobs=4)   # roc_auc, balanced_accuracy, f1
    gsearch.fit(X_train, y_train)
    print('Best parameters for recall:', gsearch.best_params_)
    print('Best score for recall:', gsearch.best_score_)

    # plot feature importance
    ax = lgb.plot_importance(gsearch, height = 0.4,
                             max_num_features = 25,
                             xlim = (0,100), ylim = (0,23),
                             figsize = (10,6))
    plt.show()
    '''
    return optimal_param


def train_xgbt_model(X_train, X_valid, y_train, y_valid):
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
    # model parameter
    model = xgb.XGBClassifier(
                           tree_method=['gpu_hist'],
                           max_depth=7,
                           min_child_weight=1,
                           learning_rate=0.1,
                           n_estimators=500,
                           silent=True,
                           objective='binary:logistic',
                           gamma=0,
                           max_delta_step=0,
                           subsample=1,
                           colsample_bytree=1,
                           colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=20,     # address imbalanced label
                           seed=1,
                           missing=None)

    # training paramter
    model.fit(X_train, y_train, eval_metric=['error', 'auc'], verbose=True,
            eval_set=[(X_valid, y_valid)], early_stopping_rounds=10)
    return model


def pred_data(model, test_dir, pred_dir, result_dir, column_mean_dict, scaler):
    test_files = glob.glob(test_dir + '*.psv')
    n_pred_pos = 0
    n_true_pos = 0
    n_file = 0
    my_pred = np.array([])
    my_label = np.array([])

    print('Predicting...')
    for file_name in test_files:
        n_file += 1
        if n_file % 1000 == 0:
            print(file_name)
            print(n_file)

        if file_name.endswith('.psv'):
            # print('predict ' + file_name)
            (values, column_names) = read_challenge_data(file_name)
            df = pd.DataFrame(values, columns=column_names)
            # impute missing value by forward and backward filling in
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df = df.fillna(value=column_mean_dict)

            # scale the data
            df[column_names[:-1]] = scaler.fit_transform(df[column_names[:-1]])

            # drop columns with high missing ratio
            # df = df.drop(dropped_columns, axis=1)

            y_prob = model.predict(df.values[:, :-1])  # probability
            y_true = df.values[:, -1]
            y_pred = y_prob > 0.5

            n_pred_pos += np.sum(y_pred)
            n_true_pos += np.sum(df.values[:, -1])
            my_pred = np.concatenate((my_pred, y_pred), axis=0) if my_pred.size else y_pred
            my_label = np.concatenate((my_label, y_true), axis=0) if my_pred.size else y_true

            # write predictions to output file
            output_file = pred_dir + file_name[-10:]
            with open(output_file, 'w') as f:
                f.write('PredictedProbability|PredictedLabel\n')
                for (s, l) in zip(y_prob, y_pred):
                    f.write('%g|%d\n' % (s, l))

    print('Predicted positive:', n_pred_pos)
    print('Actual positive:', n_true_pos)
    print(classification_report(my_label, my_pred, target_names=['neg', 'pos']))

    # compute score
    auroc, auprc, accuracy, f_measure, utility = compute_scores_2019(test_dir, pred_dir)
    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}\n'.format(auroc, auprc, accuracy, f_measure, utility)
    print(output_string)

    # save results to file
    result_file = result_dir + 'utility_' + str(utility) + '_xgbt.txt'
    if result_file:
        with open(result_file, 'w+') as f:
            f.write(output_string)

    # save model to file
    pickle.dump(model, open(result_dir + 'utility_' + str(utility) + '_xgbt.pickle', "wb"))


def run_xgbt_model(mode):
    file_name = 'D:/Projects/physionet-master/training_data/raw_stage1_data.csv'
    test_dir = 'D:/Projects/physionet-master/test_dir/'
    pred_dir = 'D:/Projects/physionet-master/pred_dir/'
    result_dir = 'D:/Projects/physionet-master/result_dir/'
    column_mean_dict = read_pickle('column_mean.pickle')
    scaler = load_scaler('maxmin_scaler.save')

    n_record = 1740663
    n_train = 1357085  # 1357085
    n_test = n_record - n_train

    X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_data(file_name, n_record, n_train, n_test)
    if mode == 'tune':
        tune_xgbt_model(X_train, X_valid, y_train, y_valid)
    if mode == 'train':
        model = train_xgbt_model(X_train, X_valid, y_train, y_valid)
        pred_data(model, test_dir, pred_dir, result_dir, column_mean_dict, scaler)


if __name__ == '__main__':
    mode = 'tune'
    run_xgbt_model(mode)


