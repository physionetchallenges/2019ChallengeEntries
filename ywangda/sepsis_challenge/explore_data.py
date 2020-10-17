import time
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from dateutil.parser import parse
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from pylab import mpl
from scipy import stats
from scipy.stats import norm, skew



import warnings
def ignore_warn(*args , **kwargs):
    pass
warnings.warn = ignore_warn
pd.set_option('display.float_format',lambda x:'{:.3f}'.format(x))    # 控制输出为精确到小数点
color = sns.color_palette()
sns.set_style('darkgrid')



#################################################################################################
#Step1： count missing value
#################################################################################################


def stat_missing_value():
    data_path = 'D:/Projects/physionet-master/training_data/'
    data = pd.read_csv(data_path + 'combined_data.csv')
    data = data.iloc[1:-1]
    missing_ratio = data.isnull().sum()/len(data)
    print ('missing ratio:', missing_ratio)
    print (missing_ratio)

    null_percentage = missing_ratio.reset_index()
    null_percentage.columns = ['column_name', 'column_value']
    ind = np.arange(null_percentage.shape[0])
    fig, ax = plt.subplots(figsize = (8, 5))
    rects = ax.barh(ind, null_percentage.column_value.values, color='blue')
    ax.set_yticks(ind)
    ax.set_yticklabels(null_percentage.column_name.values, rotation='horizontal')
    ax.set_xlabel("missing ratio of each column")
    plt.show()

    column_names = data.columns.values[1:]
    X, y = data[column_names[:-1]].values, data[column_names[-1]].values


#################################################################################################
# step2：sepsis statistic
#################################################################################################
'''
Output:
patient_num: 5000
sepesis_num: 279
train_sepsis_num: 227
test_sepsis_num: 52
train record: 150596
total record: 188453
'''
import glob
import pandas as pd
import numpy as np


def get_columns_names(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_name = header.split('|')
    return column_name


def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_name = header.split('|')
        value = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present
    '''
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    '''
    return value, column_name


def stat_sepsis_label():
    data_path = 'D:/Projects/physionet-master/training_data/'
    files = glob.glob(data_path + '*.psv')
    train_num = int(0.8*len(files))
    test_num = len(files) - train_num

    sepsis_df = pd.read_csv(data_path + 'sepsis_profile.csv')
    column_names = sepsis_df.columns.values
    sepsis_label = sepsis_df.iloc[:, -1].values
    patient_num = len(sepsis_label)
    sepesis_num = np.sum(sepsis_label)
    train_sepsis_num = np.sum(sepsis_label[:train_num])
    test_sepsis_num = np.sum(sepsis_label[train_num:])
    print('patient_num:', patient_num)
    print('sepesis_num:', sepesis_num)
    print('train_sepsis_num:', train_sepsis_num)
    print('test_sepsis_num:', test_sepsis_num)
    # shuffle is not required

    file_num = 0
    record_num = 0
    for file_name in files:
        if file_name.endswith('.psv'):
            (values, column_names) = read_challenge_data(file_name)
            record_num += values.shape[0]
            file_num += 1
            if file_num == train_num:
                print('train record:', record_num)
    print('total record:', record_num)


#################################################################################################
#step3： compute feature importance
#################################################################################################
from compute_scores_2019 import compute_accuracy_f_measure, compute_prediction_utility, compute_auc, compute_scores_2019
from sklearn.model_selection import train_test_split
import math

def compute_feature_imp():
    data_path = 'D:/Projects/physionet-master/training_data/'
    test_dir = 'D:/Projects/physionet-master/test_dir/'
    pred_dir = 'D:/Projects/physionet-master/pred_dir/'
    result_dir = 'D:/Projects/physionet-master/result_dir/'
    imputed_df = pd.read_csv(data_path + 'imputed_scaled_data.csv')
    column_names = imputed_df.columns.values
    X, y = imputed_df[column_names[:-1]].values, imputed_df[column_names[-1]].values

    # split the data
    train_num = 150596           # records of first 4000 patients
    test_num = 188453 - 150596   # records of last 1000 patients
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_num, test_size=test_num)
    files = glob.glob(data_path + '*.psv')
    train_num = int(0.8*len(files))
    test_files = glob.glob(test_dir + '*.psv')


    # specify the xgbt
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # objective function
    def loglikelihood(preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))      # logistics function
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad, hess

    # eval metric
    # binary error
    def binary_error(preds, train_data):
        labels = train_data.get_label()
        return 'error', np.mean(labels != (preds > 0.5)), False

    '''
    # accuracy
    def accuracy(preds, train_data):
        labels = train_data.get_label()
        return 'accuracy', np.mean(labels == (preds > 0.5)), True
    '''

    # 5-fold crossvalidation
    t0 = time.time()
    train_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros((X_test.shape[0]))
    n_fold = 5
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=520)
    feat_imp = np.zeros(len(column_names[:-1]))
    i = 0

    for train_ind, valid_ind in kf.split(X_train):
        print('{}th fold training...'.format(i+1))
        lgb_train = lgb.Dataset(X_train[train_ind, :], y_train[train_ind])
        lgb_valid = lgb.Dataset(X_train[valid_ind, :], y_train[valid_ind])
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=lgb_valid,
                        verbose_eval=100,
                        fobj=loglikelihood,
                        feval=binary_error,
                        early_stopping_rounds=100)

        feat_imp += gbm.feature_importance()

        # test
        for file_name in test_files:
            if file_name.endswith('.psv'):
                (values, column_names) = read_challenge_data(file_name)
                df = pd.DataFrame(values, columns=column_names)
                # impute missing value by forward and backward filling in
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                nan_flag = df.isnull().values.any()
                y_score = gbm.predict(df.values[:, :-1], num_iteration=gbm.best_iteration)  # probability
                y_prob = 1 / (1 + np.exp(-y_score))
                y_pred = y_prob > 0.5

                # write predictions to output file
                output_file = pred_dir + file_name[-10:]
                with open(output_file, 'w') as f:
                    f.write('PredictedProbability|PredictedLabel\n')
                    for (s, l) in zip(y_prob, y_pred):
                        f.write('%g|%d\n' % (s, l))

        # compute score
        auroc, auprc, accuracy, f_measure, utility = compute_scores_2019(test_dir, pred_dir)
        output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility\n{}|{}|{}|{}|{}\n'.format(auroc, auprc, accuracy, f_measure, utility)

        # save model to file
        gbm.save_model('result_dir + lgb_model_' + str(utility) + '.txt')

        #save results to file
        result_file = result_dir + '' + str(utility) + '.txt'
        if result_file:
            with open(result_file, 'w+') as f:
                f.write(output_string)
        print(output_string)

    print('trainig time:', (time.time() - t0))
    feat_imp = pd.Series(feat_imp/5, index=column_names[:-1]).sort_values(ascending=False)
    print(feat_imp)


    # mpl.rcParams['font.sans-serif'] = ['FangSong']

    feat_imp = feat_imp.reset_index()
    feat_imp.columns = ['column_name', 'column_value']
    ind = np.arange(feat_imp.shape[0])
    fig, ax = plt.subplots(figsize=(6, 8))
    rects = ax.barh(ind, feat_imp.column_value.values, color='blue')
    ax.set_yticks(ind)
    ax.set_yticklabels(feat_imp.column_name.values, rotation='horizontal')
    ax.set_xlabel("feature importance")
    plt.show()


#################################################################################################
#step4： feature association analysis
#################################################################################################
import time
def plot_feature_label():
    data_path = 'D:/Projects/physionet-master/training_data/'
    result_dir = 'D:/Projects/physionet-master/result_dir/'
    raw_df = pd.read_csv(data_path + 'combined_data.csv')
    column_names = raw_df.columns.values[1:]
    X, y = raw_df[column_names[:-1]].values, raw_df[column_names[-1]].values

    for i in range(len(column_names[:-1])):
        mpl.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots()
        ax.scatter(x=X[:, i], y=y, s=0.2)
        plt.ylabel(column_names[-1])
        plt.xlabel(column_names[i])
        # plt.show(1)
        plt.savefig(result_dir + column_names[i] + '.png')

#################################################################################################
#step4：feature distribution
#################################################################################################

def plot_feature_distribution():
    data_path = 'D:/Projects/physionet-master/training_data/'
    result_dir = 'D:/Projects/physionet-master/result_dir/'
    imputed_df = pd.read_csv(data_path + 'combined_imputed_data.csv')
    raw_df = pd.read_csv(data_path + 'combined_data.csv')
    column_names = imputed_df.columns.values[1:]
    X, y = imputed_df[column_names[:-1]].values, imputed_df[column_names[-1]].values

    for i in range(len(column_names[:-1])):
        mpl.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots()
        feat = X[:, i]
        feat = feat[~np.isnan(feat)]
        # plot a univariate distribution of observations
        dist = sns.distplot(feat, fit=norm)
        (mu, sigma) = norm.fit(feat)
        print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
        plt.ylabel('Frequency')
        plt.title(column_names[i])
        plt.savefig(result_dir + column_names[i] + '_dist' + '.png')
        # Generates a probability plot of sample data against the quantiles of a specified theoretical distribution
        fig = plt.figure()
        res = stats.probplot(X[:, i], plot=plt)

        # plt.show()
        plt.savefig(result_dir + column_names[i] + '_probplot' +'.png')

#################################################################################################
#step5：call coxbox function to smooth data
#################################################################################################

def plot_revised_distribution():
    # Return a positive dataset transformed by a Box-Cox power transformation.
    data_path = 'D:/Projects/physionet-master/training_data/'
    result_dir = 'D:/Projects/physionet-master/result_dir/'
    imputed_df = pd.read_csv(data_path + 'combined_imputed_data.csv')
    raw_df = pd.read_csv(data_path + 'combined_data.csv')
    column_names = imputed_df.columns.values[1:]
    X, y = imputed_df[column_names[:-1]].values, imputed_df[column_names[-1]].values

    for i in range(len(column_names[:-1])):
        print('column name:', column_names[i])
        feat = X[:, i]
        feat = feat[~np.isnan(feat)]
        soft, b = stats.boxcox(feat)

        mpl.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots()
        sns.distplot(soft, fit=norm)
        (mu, sigma) = norm.fit(soft)
        print('mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
        plt.ylabel('Frequency')
        plt.title(column_names[i])
        plt.savefig(result_dir + column_names[i] + '_revised_dist' + '.png')

        fig = plt.figure()
        res = stats.probplot(soft, plot=plt)
        plt.savefig(result_dir + column_names[i] + '_revised_probplot' + '.png')


#################################################################################################
#step6: 探索数据关联性特征
#################################################################################################

def plot_correlation():
    data_path = 'D:/Projects/physionet-master/training_data/'
    imputed_df = pd.read_csv(data_path + 'combined_imputed_data.csv')
    raw_df = pd.read_csv(data_path + 'combined_data.csv')
    column_names = imputed_df.columns.values[1:]
    X, y = imputed_df[column_names[:-1]].values, imputed_df[column_names[-1]].values

    mpl.rcParams['axes.unicode_minus'] = False
    corrmat =imputed_df[column_names].corr()
    f, ax = plt.subplots(figsize=(15, 12))
    ax.set_xticklabels(corrmat, rotation='horizontal')
    sns.heatmap(corrmat, vmax =0.9, square=True)
    label_y = ax.get_yticklabels()
    plt.setp(label_y , rotation = 360)
    label_x = ax.get_xticklabels()
    plt.setp(label_x , rotation = 90)
    plt.show()


if __name__=='__main__':
    # stat_missing_value()
    # stat_sepsis_label()
    compute_feature_imp()
    # plot_feature_label()
    # plot_feature_distribution()
    # plot_revised_distribution()
    # plot_correlation()
