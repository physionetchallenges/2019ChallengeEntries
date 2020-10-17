from compute_scores_2019 import compute_accuracy_f_measure, compute_prediction_utility, compute_auc, compute_scores_2019
from explore_data import read_challenge_data, get_columns_names
from preprocess_data import *
from util import *

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

import glob
import pandas as pd
import numpy as np
import time

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier


def precision_keras(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_keras(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    precision = precision_keras(y_true, y_pred)
    recall = recall_keras(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def train_mlp_model_keras(X_train, X_valid, y_train, y_valid, column_names, column_mean_dict, scaler, result_dir):
    pos_weight = 30
    ker_init = 'he_normal'
    batch_size = 16
    activation = 'relu'
    hidden_dim = 64
    hidden_dim2 = 8
    drop_out = 0.1
    prob_thresh = 0.5
    run = 1

    # undropped_columns = [x for x in list(column_names) if x not in dropped_columns]

    print('train positive:', int(np.sum(y_train)))
    # print('Test positive:', int(np.sum(y_test)))

    input_dim = X_train.shape[1]
    from sklearn.utils import class_weight

    # build neural network
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.5))

    model.add(Dense(100, input_dim=200, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.25))

    model.add(Dense(50, input_dim=100, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.1))

    model.add(Dense(25, input_dim=50, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(0.05))

    model.add(Dense(1, input_dim=25, kernel_initializer=ker_init))
    model.add(Activation('sigmoid'))

    # optimization setting
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[recall_keras, precision_keras, f1])

    # compute class weight
    # class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    # print('class_weight:', {0.0: 0.5, 1.0: pos_weight})
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    # ModelCheckpoint(filepath=result_dir + 'pos_weight_' + str(pos_weight) + '_mlp.h5', monitor='val_loss', save_best_only=True)]
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)    # chekc localhost:6006

    model.fit(X_train, y_train, batch_size=batch_size, epochs=30, verbose=1, shuffle=True,
              class_weight={0.0: 0.5, 1.0: pos_weight}, validation_data=(X_valid, y_valid), callbacks=[tensorboard, early_stop])
    return model


def tune_mlp_model_keras(X_train, X_valid, y_train, y_valid, column_names, column_mean_dict, scaler, result_dir):
    pos_weight = 10
    ker_init = 'he_normal'
    batch_size = 16
    activation = 'relu'
    hidden_dim = 128
    hidden_dim2 = 16
    drop_out = 0.1
    prob_thresh = 0.5
    run = 1
    # undropped_columns = [x for x in list(column_names) if x not in dropped_columns]

    print('Train positive:', int(np.sum(y_train)))
    # print('Test positive:', int(np.sum(y_test)))

    input_dim = X_train.shape[1]

    # build neural network
    model = Sequential()
    model.add(Dense(hidden_dim, input_dim=input_dim, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(drop_out))

    model.add(Dense(hidden_dim2, input_dim=hidden_dim, kernel_initializer=ker_init))  #
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(drop_out))

    model.add(Dense(1, input_dim=hidden_dim2, kernel_initializer=ker_init))
    model.add(Activation('sigmoid'))

    # optimization setting
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[recall_keras, recall_keras, precision_keras, f1])
    model = KerasClassifier(model)
    # compute class weight
    # class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    # print('class_weight:', {0.0: 0.5, 1.0: pos_weight})
    early_stop = [EarlyStopping(monitor='val_loss', patience=2)]
                 # ModelCheckpoint(filepath=result_dir + 'pos_weight_' + str(pos_weight) + '_mlp.h5, monitor='val_loss', save_best_only=True)]

    # model.fit(X_train, y_train, nb_epoch=30, verbose=2, batch_size=batch_size, shuffle=True,
              # class_weight={0.0: 0.5, 1.0: pos_weight}, validation_data=(X_valid, y_valid), callbacks=callbacks)

    param_grid1 = {
        'batch_size': [16, 64, 128],
    }
    '''
    param_grid2 = {
        'batch_size': [16, 64, 128],
        'epochs': [10, 50, 100]
    }

    param_grid3 = {
        'batch_size': [16, 64, 128],
        'epochs': [10, 50, 100]
    }

    param_grid4 = {
        'batch_size': [16, 64, 128],
        'epochs': [10, 50, 100]
    }
    '''
    optimal_param = {}
    for param_grid in [param_grid1]:
        # Create a TensorBoard instance with the path to the logs directory
        tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0,
                                    write_graph=True, write_images=True)

        # get found optimal parameters
        param_grid.update(optimal_param)
        gsearch = GridSearchCV(estimator=model, param_grid=param_grid, scoring='balanced_accuracy', cv=5, verbose=1,
                               n_jobs=4)  # roc_auc, balanced_accuracy, f1
        gsearch.fit(X_train, y_train, class_weight={0.0: 0.5, 1.0: pos_weight}, validation_data=(X_valid, y_valid), callbacks=[early_stop, tensorboard])
        print('Best parameters found by grid search are:', gsearch.best_params_)
        print('Best score found by grid search are:', gsearch.best_score_)

        # change dict values format
        best_params = gsearch.best_params_
        for k in best_params:
            best_params[k] = [best_params[k]]      # put int in a list

        # update optimal param
        optimal_param.update(best_params)
    return optimal_param

'''
def tune_mlp_model_sk(para, X_train, X_valid, y_train, y_valid, column_names, column_mean_dict, scaler, result_dir):
    pos_weight = 10
    ker_init = 'he_normal'
    batch_size = 16
    activation = 'relu'
    hidden_dim = 64
    hidden_dim2 = 16
    drop_out = 0.1
    prob_thresh = para
    run = para

    print('Train positive:', int(np.sum(y_train)))
    # print('Test positive:', int(np.sum(y_test)))

    input_dim = X_train.shape[1]

    # build neural network
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.0001,
                        solver='sgd', verbose=10, random_state=21, tol=0.000000001)

    #mlp.fit(X_train, y_train)
    #y_pred = mlp.predict(X_test)

    param_grid1 = {
        'hidden_layer_sizes': [(100, 100, 100), (100, 100), 128],
        'epochs': [10, 50, 100]

    }

    param_grid2 = {
        'batch_size': [16, 64, 128],
        'epochs': [10, 50, 100]
    }

    param_grid3 = {
        'batch_size': [16, 64, 128],
        'epochs': [10, 50, 100]
    }

    param_grid4 = {
        'batch_size': [16, 64, 128],
        'epochs': [10, 50, 100]
    }

    optimal_param = {}
    for param_grid in [param_grid1, param_grid2, param_grid3, param_grid4]:
        # Create a TensorBoard instance with the path to the logs directory
        tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

        # get found optimal parameters
        param_grid.update(optimal_param)
        gsearch = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='balanced_accuracy', cv=5, verbose=1,
                               n_jobs=4)  # roc_auc, balanced_accuracy, f1
        gsearch.fit(X_train, y_train)
        print('Best parameters found by grid search are:', gsearch.best_params_)
        print('Best score found by grid search are:', gsearch.best_score_)

        # change dict values format
        best_params = gsearch.best_params_
        for k in best_params:
            best_params[k] = [best_params[k]]      # put int in a list

        # update optimal param
        optimal_param.update(best_params)
    return optimal_param
'''

def pred_data(model, test_dir, pred_dir, result_dir, feat_columns, column_mean_dict, scaler):
    print('predicting...')
    test_files = glob.glob(test_dir + '*.psv')

    n_pred_pos = 0
    n_true_pos = 0
    n_file = 0
    my_pred = np.array([])
    my_label = np.array([])

    for file_name in test_files:
        if file_name.endswith('.psv'):
            # print('predict ' + file_name)
            (values, column_names) = read_challenge_data(file_name)
            df = pd.DataFrame(values, columns=column_names)

            # preprocess the predicted input
            df = preprocess_pred(df, column_mean_dict)

            # df = df.drop(dropped_columns, axis=1)     # drop columns with high missing ratio
            y_true = df['SepsisLabel'].values
            X_test = df[feat_columns].values
            y_prob = model.predict_proba(X_test)[:, 1]  # positive probability
            y_pred = y_prob > 0.3     # key parameter

            n_pred_pos += np.sum(y_pred)
            n_true_pos += np.sum(y_true)
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

    # save model weights
    model.save_weights(result_dir + 'utility_' + str(utility) + '_mlp.h5') # save the model

    # save results to file
    result_file = result_dir + 'utility_' + str(utility) + '_mlp.txt'

    if result_file:
        with open(result_file, 'w+') as f:
            f.write(output_string)


def run_mlp_model(mode):
    # file_name = 'D:/Projects/physionet-master/training_data/aug_imputed_stage1_data.csv'
    # = 'D:/Projects/physionet-master/training_data/imputed_stage1_data.csv'
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

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(file_name, n_record, n_train, n_test)
    if mode == 'tune':
        tune_mlp_model_keras(X_train, X_valid, y_train, y_valid, column_names, column_mean_dict, scaler, result_dir)
    if mode == 'train':
        model = train_mlp_model_keras(X_train, X_valid, y_train, y_valid, column_names, column_mean_dict, scaler, result_dir)
        pred_data(model, test_dir, pred_dir, result_dir, feat_columns, column_mean_dict, scaler)


if __name__ == '__main__':
    mode = 'train'
    run_mlp_model(mode)