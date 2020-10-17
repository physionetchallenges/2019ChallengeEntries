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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve


def train_lr_model(X_train, X_valid, y_train, y_valid):
    # sklearn API
    # init
    lr = LogisticRegression(penalty='l2',
                            class_weight='balanced',
                            random_state=0,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='ovr').fit(X_train, y_train)
    sparsity = np.mean(lr.coef_ == 0) * 100
    return lr


def tune_lr_model(X_train, X_valid, y_train, y_valid):
    print('Tuning by sklearn API...')
    lr = LogisticRegression(n_jobs=4)

    # tune parameters
    param_grid1 = {
        'penalty': ['l2', 'l1'],                  # check no whitespace
        'C': [1, 0.8, 0.5]
    }

    param_grid2 = {
        'class_weight': [{0: 0.5, 1: 10}, {0: 0.5, 1: 30}, {0: 0.5, 1: 50}]
    }

    param_grid3 = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }

    optimal_param = {}
    for param_grid in [param_grid1, param_grid2, param_grid3]:
        # get found optimal parameters
        param_grid.update(optimal_param)

        # score by balanced_accuracy
        gsearch = GridSearchCV(estimator=lr,
                               param_grid=param_grid,
                               scoring='balanced_accuracy',
                               cv=5,
                               verbose=1,
                               n_jobs=4)                                # roc_auc, balanced_accuracy, f1
        gsearch.fit(X_train, y_train)
        print('Best parameters found by grid search are:', gsearch.best_params_)
        print('Best score found by grid search are:', gsearch.best_score_)

        # change dict values format
        best_params = gsearch.best_params_
        for k in best_params:
            best_params[k] = [best_params[k]]  # put int in a list

        # update optimal param
        optimal_param.update(best_params)

        return optimal_param


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
    pickle.dump(model, open(result_dir + 'utility_' + str(utility) + '_lr.pickle', "wb"))

    # save results to file
    result_file = result_dir + 'utility_' + str(utility) + '_lr.txt'
    if result_file:
        with open(result_file, 'w+') as f:
            f.write(output_string)


def run_lr_model(mode):
    data_path = 'D:/Projects/physionet-master/training_data/'
    # file_name = 'D:/Projects/physionet-master/training_data/raw_stage1_data.csv'
    # file_name = 'D:/Projects/physionet-master/training_data/imputed_stage1_data.csv'            # imputing is important
    file_name = data_path + 'aug_imputed_stage1_data.csv'       # imputing is important
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
        tune_lr_model(X_train, X_valid, y_train, y_valid)
    if mode == 'train':
        model = train_lr_model(X_train, X_valid, y_train, y_valid)
        pred_data(model, test_dir, pred_dir, result_dir, feat_columns, column_mean_dict, scaler)


if __name__ == '__main__':
    mode = 'tune'                # train or tune
    run_lr_model(mode)
