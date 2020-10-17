# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import glob
from compute_scores_2019 import compute_accuracy_f_measure, compute_prediction_utility, compute_auc, compute_scores_2019
from explore_data import read_challenge_data, get_columns_names
from util import *
from preprocess_data import *
from sklearn.metrics import precision_recall_curve


def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess


def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False


def accuracy(preds, train_data):
    labels = train_data.get_label()
    return 'accuracy', np.mean(labels == (preds > 0.5)), True


def select_feature(X, selected_columns):
    X_reduce = X[:, selected_columns]

    return X_reduce


def generate_feature(X):
    # log feature
    X[:, -2] = -X[:,  -2]  # convert 'HospAdmTime' to non-negative
    feat_dim = X.shape[1]
    for i in range(feat_dim):
        for j in range(i, feat_dim):
            c = X[:, [i, j]]
            pl = PolynomialFeatures(2)
            b = pl.fit_transform(c)
            # only add second moment feature
            X = np.concatenate((X, b[:, 3:]), axis=1)
    return X


def tune_lgb_model(X_train, X_valid, y_train, y_valid):
    # sklearn API
    # init
    print('Tuning by sklearn API...')
    gbm = lgb.LGBMClassifier(objective='binary',
                             learning_rate=0.1,
                             n_estimators=20,
                             metric='binary_logloss',
                             # bagging_fraction=0.8,
                             # feature_fraction=0.8,
                             num_leaves=71,
                             max_depth=6,
                             subsample_for_bin=200000,
                             class_weight={0: 0.5, 1: 20},
                             min_split_gain=0.0,
                             min_child_weight=0.001,
                             min_child_samples=30,
                             subsample=0.6,
                             subsample_freq=0,
                             colsample_bytree=0.9,
                             reg_alpha=0.01,
                             reg_lambda=0.0,
                             random_state=None,
                             n_jobs=4,
                             silent=True)

    # tune parameters
    param_grid1 = {
        'n_estimators': [20, 100, 200, 500],
        'max_depth': [5, 6, 7, 8],
    }

    param_grid2 = {
        'class_weight': [{0: 0.5, 1: 10}, {0: 0.5, 1: 20}, {0: 0.5, 1: 30}, {0: 0.5, 1: 40}, {0: 0.5, 1: 50}]
    }

    param_grid3 = {
        'min_child_samples': [10, 30, 50, 100],
        'min_child_weight': [0.001, 0.01, 0.1]
    }

    param_grid4 = {
        'subsample': [0.6, 0.8, 1],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    param_grid5 = {
        'reg_alpha': [0, 0.001, 0.01, 0.1],
        'reg_lambda': [0, 0.001, 0.01, 0.1]
    }

    # score by recall
    '''
    gsearch = GridSearchCV(estimator=gbm, param_grid=param_grid, scoring='recall', cv=5, verbose=1, n_jobs=4)   # roc_auc, balanced_accuracy, f1
    gsearch.fit(X_train, y_train)
    print('Best parameters found by grid search are:', gsearch.best_params_)
    print('Best score found by grid search are:', gsearch.best_score_)
    '''

    optimal_param = {}
    for param_grid in [param_grid1, param_grid2, param_grid3, param_grid4, param_grid5]:
        # get found optimal parameters
        param_grid.update(optimal_param)

        # score by balanced_accuracy
        gsearch = GridSearchCV(estimator=gbm, param_grid=param_grid, scoring='balanced_accuracy', cv=5, verbose=1, n_jobs=4)  # roc_auc, balanced_accuracy, f1
        gsearch.fit(X_train, y_train)
        print('Best parameters found by grid search are:', gsearch.best_params_)
        print('Best score found by grid search are:', gsearch.best_score_)

        # change dict values format
        best_params = gsearch.best_params_
        for k in best_params:
            best_params[k] = [best_params[k]]      # put int in a list

        # update optimal param
        optimal_param.update(best_params)


    '''
    # score by recall
    gsearch = GridSearchCV(estimator=gbm, param_grid=param_grid, scoring='recall', cv=5, verbose=1, n_jobs=4)   # roc_auc, balanced_accuracy, f1
    gsearch.fit(X_train, y_train)
    print('Best parameters found by grid search are:', gsearch.best_params_)
    print('Best score found by grid search are:', gsearch.best_score_)


    # f1_weighted
    gsearch = GridSearchCV(estimator=gbm, param_grid=param_grid, scoring='f1_weighted', cv=5, verbose=1, n_jobs=4)   # roc_auc, balanced_accuracy, f1
    gsearch.fit(X_train, y_train)
    print('Best parameters found by grid search are:', gsearch.best_params_)
    print('Best score found by grid search are:', gsearch.best_score_)
    '''

    '''
    # plot feature importance
    ax = lgb.plot_importance(gsearch, height = 0.4,
                             max_num_features = 25,
                             xlim = (0,100), ylim = (0,23),
                             figsize = (10,6))
    plt.show()
    '''
    return optimal_param


def train_lgb_model_sk(X_train, X_valid, y_train, y_valid):
    # sklearn API
    # init
    print('Training by sklearn API...')
    gbm = lgb.LGBMClassifier(objective='binary',
                             learning_rate=0.1,
                             n_estimators=100,       # 100
                             metric='binary_logloss',
                             num_leaves=71,
                             max_depth=5,            # 6
                             subsample_for_bin=200000,
                             class_weight={0: 0.5, 1: 30},
                             min_split_gain=0.0,
                             min_child_weight=0.001,
                             min_child_samples=30,
                             subsample=0.6,
                             subsample_freq=0,
                             colsample_bytree=0.8,
                             reg_alpha=0.01,
                             reg_lambda=0.001,
                             random_state=None,
                             n_jobs=4,
                             silent=True)

    # train
    gbm.fit(X_train, y_train)
    return gbm


def train_lgb_model_py(X_train, X_valid, y_train, y_valid):
    # python API
    # create dataset for lightgbm
    print('Tuning by python API...')
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    # init parameters
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_boost_round': 100,
        'learning_rate': 0.01,
        'device': 'cpu',
        'nthread': 4,
        'max_depth':  6,
        'num_leaves': 51,
        'scale_pos_weight': 20,     # imbalanced data
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    fobj=loglikelihood,
                    feval=binary_error,
                    valid_sets=lgb_valid,
                    early_stopping_rounds=10)

    return gbm


def pred_data(model, test_dir, pred_dir, result_dir, feat_columns, column_mean_dict, scaler):
    print('Predicting...')
    t = time.time()
    test_files = glob.glob(test_dir + '*.psv')
    n_pred_pos = 0
    n_true_pos = 0
    my_pred = np.array([])
    my_label = np.array([])

    for file_name in test_files:
        if file_name.endswith('.psv'):
            # print('predict ' + file_name)
            (values, column_names) = read_challenge_data(file_name)
            df = pd.DataFrame(values, columns=column_names)
            # preprocess predicted data
            df = preprocess_pred(df, column_mean_dict)

            # scale the data
            # df[column_names[:-1]] = scaler.fit_transform(df[column_names[:-1]])

            # drop columns with high missing ratio
            # df = df.drop(dropped_columns, axis=1)

            y_true = df['SepsisLabel'].values
            X_test = df[feat_columns].values
            y_prob = model.predict_proba(X_test)[:, 1]  # positive probability
            y_pred = y_prob > 0.5

            my_pred = np.concatenate((my_pred, y_pred), axis=0) if my_pred.size else y_pred
            my_label = np.concatenate((my_label, y_true), axis=0) if my_pred.size else y_true
            n_true_pos += np.sum(y_true)
            n_pred_pos += np.sum(y_pred)

            # write predictions to output file
            output_file = pred_dir + file_name[-10:]
            with open(output_file, 'w') as f:
                f.write('PredictedProbability|PredictedLabel\n')
                for (s, l) in zip(y_prob, y_pred):
                    f.write('%g|%d\n' % (s, l))

    print(classification_report(my_label, my_pred, target_names=['neg', 'pos']))
    print('Predicted positive:', n_pred_pos)
    print('True positive:', n_true_pos)
    print('Elapsed time: ', time.time() - t)

    # compute score
    auroc, auprc, accuracy, f_measure, utility = compute_scores_2019(test_dir, pred_dir)
    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}\n'.format(auroc, auprc, accuracy, f_measure, utility)
    print(output_string)

    # save model weights
    # model.save_weights(result_dir + 'utility_' + str(utility) + '_' + saved_model_file) # save the model
    # save model to file
    pickle.dump(model, open(result_dir + 'utility_' + str(utility) + '_lgb.pickle', "wb"))

    # save results to file
    result_file = result_dir + 'utility_' + str(utility) + '_lgb.txt'
    if result_file:
        with open(result_file, 'w+') as f:
            f.write(output_string)


def run_lgb_model(mode):
    data_path = 'D:/Projects/physionet-master/training_data/'
    # file_name = 'D:/Projects/physionet-master/training_data/raw_stage1_data.csv'
    # file_name = 'D:/Projects/physionet-master/training_data/imputed_stage1_data.csv'            # imputing is important
    file_name = data_path + 'aug_imputed_stage1_data.csv'        # imputing is important
    test_dir = 'D:/Projects/physionet-master/test_dir/'
    pred_dir = 'D:/Projects/physionet-master/pred_dir/'
    result_dir = 'D:/Projects/physionet-master/result_dir/'
    column_mean_dict = read_pickle('aug_column_mean.pickle')
    scaler = load_scaler('maxmin_scaler.save')

    n_record = 1740663
    n_train = 1357085  # 1357085
    n_test = n_record - n_train
    column_names = pd.read_csv('D:/Projects/physionet-master/aug_column_mean.csv').columns.values
    feat_columns = np.delete(column_names, np.where(column_names == 'SepsisLabel'))

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(file_name, n_record, n_train, n_test)

    if mode == 'tune':
        tune_lgb_model(X_train, X_valid, y_train, y_valid)
    if mode == 'train':
        model = train_lgb_model_sk(X_train, X_valid, y_train, y_valid)
        pred_data(model, test_dir, pred_dir, result_dir, feat_columns, column_mean_dict, scaler)

        # plot feature importance
        ax = lgb.plot_importance(model, height=0.4,
                                 max_num_features=40,
                                 xlim=(0, 400), ylim=(0, 39),
                                 figsize=(10, 6))
        plt.show()


if __name__ == '__main__':
    mode = 'train'                # train or tune
    run_lgb_model(mode)
