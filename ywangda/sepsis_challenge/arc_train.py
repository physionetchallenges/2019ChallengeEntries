import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TerminateOnNaN, LambdaCallback

import architect
from metric import *
from cos_scheduler import *
from compute_scores_2019 import compute_accuracy_f_measure, compute_prediction_utility, compute_auc, compute_scores_2019
from explore_data import read_challenge_data, get_columns_names
from preprocess_data import *
from util import *
from sklearn.utils import class_weight

arch_names = architect.__dict__.keys()

def precision_keras(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[:, 1], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_keras(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:, 1], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    precision = precision_keras(y_true, y_pred)
    recall = recall_keras(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='mlp_arcface',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: vgg8)')
    parser.add_argument('--num-features', default=3, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--scheduler', default='CosineAnnealing',
                        choices=['CosineAnnealing', 'None'],
                        help='scheduler: ' +
                            ' | '.join(['CosineAnnealing', 'None']) +
                            ' (default: CosineAnnealing)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float)
    args = parser.parse_args()
    return args


def preprocess():
    file_name = 'D:/Projects/physionet-master/training_data/imputed_scaled_stage1_data.csv'
    project_path = 'D:/Projects/physionet-master/'
    data_path = 'D:/Projects/physionet-master/training_data/'
    # imputed_scaled_df = pd.read_csv(data_path + 'aug_imputed_scaled_stage1_data.csv')

    test_dir = project_path + 'test_dir/'
    pred_dir = project_path + 'pred_dir/'
    result_dir = project_path + 'result_dir/'
    column_names = pd.read_csv('D:/Projects/physionet-master/aug_column_mean.csv').columns.values
    feat_columns = np.delete(column_names, np.where(column_names == 'SpesisLabel'))
    '''
    dropped_columns = ['EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'Alkalinephos', 'Chloride',
                       'Bilirubin_direct', 'Lactate', 'Bilirubin_total', 'TroponinI', 'PTT', 'Fibrinogen']
    '''
    column_mean_dict = read_pickle('aug_column_mean.pickle')
    scaler = load_scaler('maxmin_scaler.save')

    n_record = 1740663
    n_train = 1357085  # 1357085
    n_test = n_record - n_train

    X, X_valid, X_test, y, y_valid, y_test = split_data(file_name, n_record, n_train, n_test)

    # convert to one hot encoding
    y = keras.utils.to_categorical(y, 2)
    y_valid = keras.utils.to_categorical(y_valid, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    return X, X_valid, X_test, y, y_valid, y_test


def main():
    args = parse_args()

    # add model name to args
    args.name = 'mnist_%s_%dd' %(args.arch, args.num_features)
    os.makedirs('models/%s' % args.name, exist_ok=True)
    print('---------------------------------------------------------------------------')
    print('config...')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('---------------------------------------------------------------------------')

    # save args
    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    # split data
    t = time.time()
    X, X_valid, X_test, y, y_valid, y_test = preprocess()
    print('preprocess time:', time.time() - t)

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr)

    print(architect.__dict__)
    model = architect.__dict__[args.arch](args)       # call archietect function
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', recall_keras, precision_keras, f1])
    model.summary()


    callbacks = [
        ModelCheckpoint(os.path.join('models', args.name, 'model.hdf5'),
            verbose=1, save_best_only=True),
        CSVLogger(os.path.join('models', args.name, 'log.csv')),
        TerminateOnNaN()]


    # change learning rate
    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))

    # handle imbalanced issue
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y[:, 1]), y[:, 1])

    # train the model
    if 'face' in args.arch:
        # callbacks.append(LambdaCallback(on_batch_end=lambda batch, logs: print('W has nan value!!') if np.sum(np.isnan(model.layers[-4].get_weights()[0])) > 0 else 0))
        model.fit([X, y], y, validation_data=([X_test, y_test], y_test),
            batch_size=args.batch_size,
            shuffle = True,
            class_weight = class_weights,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)
    else:
        model.fit(X, y, validation_data=(X_test, y_test),
            batch_size=args.batch_size,
            shuffle=True,
            class_weight = class_weights,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)


    model.load_weights(os.path.join('models/%s/model.hdf5' % args.name))
    if 'face' in args.arch:
        score = model.evaluate([X_test, y_test], y_test, verbose=1)
    else:
        score = model.evaluate(X_test, y_test, verbose=1)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == '__main__':
    main()