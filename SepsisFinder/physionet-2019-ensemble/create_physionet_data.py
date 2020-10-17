# -*- coding: utf-8 -*-
# @Author: Chloe
# @Date:   2019-07-24 16:57:51
# @Last Modified by:   Chloe
# @Last Modified time: 2019-07-25 16:02:14

import argparse
import numpy as np
import pandas as pd
import os
import time
import pickle
from dataset import PhysionetDataset, FEATURES, LABEL, LABS_VITALS

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
    train_dataset = PhysionetDataset(args.train_dir)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Preprocessing train data")
    train_dataset.__preprocess__(method=args.preprocessing_method)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Loading valid data")
    valid_dataset = PhysionetDataset(args.valid_dir)
    print("Time elapsed since start: {}".format(time.time() - start_time))
    print("Preprocessing valid data")
    valid_dataset.__preprocess__(method=args.preprocessing_method)
    print("Time elapsed since start: {}".format(time.time() - start_time))

    print("Save preprocessed datasets")

    train_filename = os.path.join(args.output_dir,
                                  "train_preprocessed_{}.pkl".format(args.preprocessing_method))
    valid_filename = os.path.join(args.output_dir,
                                  "valid_preprocessed_{}.pkl".format(args.preprocessing_method))

    with open(train_filename, "wb") as f:
        pickle.dump(train_dataset.data, file=f)
    with open(valid_filename, "wb") as f:
        pickle.dump(valid_dataset.data, file=f)
    print("Time elapsed since start: {}".format(time.time() - start_time))
