# -*- coding: utf-8 -*-
# @Author: Chloe
# @Date:   2019-07-24 16:57:51
# @Last Modified by:   Chloe
# @Last Modified time: 2019-07-24 17:43:47

import argparse
import numpy as np
import pandas as pd
import os
import time
from dataset import PhysionetDatasetCNN, FEATURES, LABEL, LABS_VITALS

if __name__ == "__main__":
    start_time = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--valid_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--window_size", default=24, type=int)
    argparser.add_argument("--output_dir", default=".")
    argparser.add_argument("--preprocessing_method", default="measured",
                           help="""Possible values:
                           - measured (forward-fill, add indicator variables, normalize, impute with -1),
                           - simple (only do forward-fill and impute with -1) """)
    args = argparser.parse_args()
    print(args)

    window_size = args.window_size
    num_features = len(FEATURES) + len(LABS_VITALS)

    print("Loading train data")
    train_dataset = PhysionetDatasetCNN(args.train_dir)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Preprocessing train data")
    train_dataset.__preprocess__(method=args.preprocessing_method)
    train_dataset.__setwindow__(window_size)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Loading valid data")
    valid_dataset = PhysionetDatasetCNN(args.valid_dir)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Preprocessing valid data")
    valid_dataset.__preprocess__(method=args.preprocessing_method)
    valid_dataset.__setwindow__(window_size)
    print("Time elapsed since start: {}".format(time.time() - start_time))

    print("Save preprocessed datasets")

    ratio_values = [None, 0.1, 0.2, 0.3, 0.4, 0.5]
    for ratio in ratio_values:
        print("Generating train data with ratio {}".format(ratio))
        if ratio:
            indices_no_outcome_keep = np.random.permutation(train_dataset.indices_no_outcome)[:int(1 / ratio) * len(train_dataset.indices_outcome)]
            indices_train = np.random.permutation(np.concatenate((train_dataset.indices_outcome, indices_no_outcome_keep)))
            train_n = len(indices_train)
        else:
            train_n = train_dataset.__len__()
            indices_train = np.concatenate((train_dataset.indices_outcome,
                                            train_dataset.indices_no_outcome))
        train_features = np.zeros((train_n, window_size, num_features))
        train_outcomes = np.zeros((train_n, 1))
        train_ids = np.zeros((train_n, 1))
        train_iculos = np.zeros((train_n, 1))
        train_filenames = np.empty((train_n), dtype="S10")

        for i in range(len(indices_train)):
            item = train_dataset.__getitem__(indices_train[i])
            train_features[i, :, :] = item[0]
            train_outcomes[i, :] = item[1]
            train_ids[i] = item[2]
            train_iculos[i] = item[3]
            train_filenames[i] = item[4]

        if ratio:
            train_filename = os.path.join(args.output_dir,
                                      "train_preprocessed_{}_window_{}_ratio_{}.npz".format(args.preprocessing_method, window_size, str(ratio).replace(".", "_")))
        else:
            train_filename = os.path.join(args.output_dir,
                                          "train_preprocessed_{}_window_{}.npz".format(args.preprocessing_method, window_size))
        np.savez(train_filename,
                 train_features=train_features,
                 train_outcomes=train_outcomes,
                 train_ids=train_ids,
                 train_iculos=train_iculos,
                 train_filenames=train_filenames)
        print("Time elapsed since start: {}".format(time.time() - start_time))

    print("Generating valid dataset")
    valid_n = valid_dataset.__len__()
    valid_features = np.zeros((valid_n, window_size, num_features))
    valid_outcomes = np.zeros((valid_n, 1))
    valid_ids = np.zeros((valid_n, 1))
    valid_iculos = np.zeros((valid_n, 1))
    valid_filenames = np.empty((valid_n), dtype="S10")
    for i in range(valid_n):
        item = valid_dataset.__getitem__(i)
        valid_features[i, :, :] = item[0]
        valid_outcomes[i, :] = item[1]
        valid_ids[i] = item[2]
        valid_iculos[i] = item[3]
        valid_filenames[i] = item[4]

    valid_filename = os.path.join(args.output_dir,
                                  "valid_preprocessed_{}_window_{}.npz".format(args.preprocessing_method, window_size))
    np.savez(valid_filename,
             valid_features=valid_features,
             valid_outcomes=valid_outcomes,
             valid_ids=valid_ids,
             valid_iculos=valid_iculos,
             valid_filenames=valid_filenames)
    print("Time elapsed since start: {}".format(time.time() - start_time))
