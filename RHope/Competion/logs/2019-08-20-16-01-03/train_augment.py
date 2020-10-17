import numpy as np
import pandas as pd
import logging
import os
import datetime
import shutil

from utils.path_utils import project_root

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from pytorch_classifier import PytorchClassifer
from lgbm_classifier import LGBMClassifier, save_features_importance, save_model, lgb_classifier_params
from compute_scores import normalized_utility_score
from config import nn_config
from tensorboardX import SummaryWriter


def log(message: str='{}', value: any=None):
    print(message.format(value))
    logging.info(message.format(value))


def initialize_local_experiment():

    exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_path = setup_local_results(exp_time)

    # data_name = 'training_filled.pickle'
    # log(message="Data name: {}", value=data_name)

    # training_examples = pd.read_pickle(os.path.join(project_root(), 'data', 'processed', data_name))
    # with open(os.path.join(project_root(), 'data', 'processed', 'lengths_all.txt')) as f:
    #     lengths_list = [int(l) for l in f.read().splitlines()]
    # with open(os.path.join(project_root(), 'data', 'processed', 'is_sepsis_all.txt')) as f:
    #     is_sepsis = [int(l) for l in f.read().splitlines()]
    training_examples = list(np.load('F:\Cinc\正式阶段\physionet-2019-challenge-master\data\Cinc_data_Augment@wang\\True_data_filled_augment_reduce_sepsis3_list.npy',allow_pickle=True))
    lengths_list      = list(np.load('F:\Cinc\正式阶段\physionet-2019-challenge-master\data\Cinc_data_Augment@wang\\True_length.npy',allow_pickle=True))
    is_sepsis         = list(np.load('F:\Cinc\正式阶段\physionet-2019-challenge-master\data\Cinc_data_Augment@wang\\True_label_list.npy',allow_pickle=True))

    training_examples2 = list(np.load('F:\Cinc\正式阶段\physionet-2019-challenge-master\data\Cinc_data_Augment@wang\\False_data_filled_augment_reduce_sepsis3_list.npy', allow_pickle=True))
    lengths_list2      = list(np.load('F:\Cinc\正式阶段\physionet-2019-challenge-master\data\Cinc_data_Augment@wang\\False_length.npy', allow_pickle=True))
    is_sepsis2         = list(np.load('F:\Cinc\正式阶段\physionet-2019-challenge-master\data\Cinc_data_Augment@wang\\False_label_list.npy', allow_pickle=True))

    training_examples.extend(training_examples2)
    lengths_list.extend(lengths_list2)
    is_sepsis.extend(is_sepsis2)

    writer = SummaryWriter(log_dir=os.path.join(project_root(), 'data', 'logs', exp_time), comment='')

    return training_examples, lengths_list, is_sepsis, writer, results_path


def setup_local_results(exp_time):

    log_root = os.path.join(project_root(), 'data', 'logs', exp_time)
    os.mkdir(log_root)
    logging.basicConfig(filename=os.path.join(log_root, exp_time + '.log'), level=logging.DEBUG)
    shutil.copy(os.path.join(project_root(), 'pytorch_classifier.py'), log_root)
    shutil.copy(os.path.join(project_root(), 'train_augment.py'), log_root)

    return log_root


def get_split(ind_train, ind_test, training_examples, lengths_list, is_sepsis):

    x_train = [t for i, t in enumerate(training_examples) if i in ind_train]
    x_train_lens = [t for i, t in enumerate(lengths_list) if i in ind_train]
    is_sepsis_train = [t for i, t in enumerate(is_sepsis) if i in ind_train]
    x_test = [t for i, t in enumerate(training_examples) if i in ind_test]
    x_test_lens = [t for i, t in enumerate(lengths_list) if i in ind_test]
    is_sepsis_test = [t for i, t in enumerate(is_sepsis) if i in ind_test]

    return x_train, x_train_lens, is_sepsis_train, x_test, x_test_lens, is_sepsis_test


def main(training_examples, lengths_list, is_sepsis, writer, results_path):
    # training_examples = list(training_examples)
    # is_sepsis= list(is_sepsis) #lengths_list
    skf = StratifiedKFold(n_splits=5)
    train_scores = []
    test_scores = []
    y_preds_test = []
    inds_test = []
    log(message="Config={}", value=nn_config)
    log(message="Config={}", value=lgb_classifier_params)
    train_scores_limit = 1000
    log(message="train_scores_limit={}", value=train_scores_limit)
    is_sepsis_ =[]
    for num in range(len(is_sepsis)):
        example = is_sepsis[num]
        is_sepsis_.append(1 if 1 in example else 0)


    for i, (ind_train, ind_test) in enumerate(skf.split(training_examples, is_sepsis_)):
        x_train, x_train_lens, is_sepsis_train, x_test, x_test_lens, is_sepsis_test = \
            get_split(ind_train, ind_test, training_examples, lengths_list, is_sepsis)

        model = LGBMClassifier(config=nn_config, writer=writer, eval_set=[(x_test, is_sepsis_test),
                                                                            x_train, is_sepsis_train])
        model.fit(x_train, x_train_lens, is_sepsis_train)
        y_pred_train, y_train = model.predict(x_train[:train_scores_limit],is_sepsis_train[:train_scores_limit])
        y_pred_test, y_test = model.predict(x_test,is_sepsis_test, search_thr=True)

        train_score, _, train_f_score = normalized_utility_score(targets=y_train[:train_scores_limit],
                                                                 predictions=y_pred_train[:train_scores_limit])
        test_score, _, test_f_score = normalized_utility_score(targets=y_test, predictions=y_pred_test)

        test_scores.append(test_score)
        train_scores.append(train_score)
        y_preds_test.extend(y_pred_test)
        inds_test.extend(list(ind_test))
        log(message="Train score: {}", value=train_score)
        log(message="Test score: {}", value=test_score)
        log(message="Train f_score: {}", value=train_f_score)
        log(message="Test f_score: {}", value=test_f_score)

        # save_features_importance(model.feature_importances_, x_train[0].columns.values,
        #                          os.path.join(results_path, 'fi.png'))

        save_model(model, path=os.path.join(results_path, 'lgbm_{}.bin'.format(i)))

    log(message="\n\nMean train MAE: {}", value=np.mean(train_scores))
    log(message="Mean test MAE: {}", value=np.mean(test_scores))
    log(message="Std train MAE: {}", value=np.std(train_scores))
    log(message="Std test MAE: {}", value=np.std(test_scores))


if __name__ == '__main__':
    training_examples, lengths_list, is_sepsis, writer, results_path = initialize_local_experiment()
    main(training_examples, lengths_list, is_sepsis, writer, results_path)
