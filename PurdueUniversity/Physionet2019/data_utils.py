#!/usr/bin/env python
import numpy as np, os, sys
import math
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from fancyimpute import SoftImpute, BiScaler

np.random.seed(7)

averages = {
    "avg_0": 85,
    "avg_1": 97.27,
    "avg_2": 37.02,
    "avg_3": 122.7,
    "avg_4": 82.4,
    "avg_5": 63.83,
    "avg_6": 18.73,
    "avg_7": 32.96,
    "avg_8": -0.69,
    "avg_9": 24.08,
    "avg_10": 0.55,
    "avg_11": 7.38,
    "avg_12": 41.02,
    "avg_13": 92.65,
    "avg_14": 260.22,
    "avg_15": 23.92,
    "avg_16": 102.48,
    "avg_17": 7.56,
    "avg_18": 105.83,
    "avg_19": 1.51,
    "avg_20": 1.84,
    "avg_21": 136.93,
    "avg_22": 2.65,
    "avg_23": 2.05,
    "avg_24": 3.54,
    "avg_25": 4.14,
    "avg_26": 2.11,
    "avg_27": 8.29,
    "avg_28": 30.79,
    "avg_29": 10.43,
    "avg_30": 41.23,
    "avg_31": 11.45,
    "avg_32": 278.39,
    "avg_33": 196.01,
    "avg_34": 62.01,
    "avg_35": 0.56,
    "avg_36": 0.5,
    "avg_37": 0.5,
    "avg_38": -57.13,
    "avg_39": 26.99
    }

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        labels, data = data[:, -1], data[:, :-1]

    return data, labels


def fill_row(data):
    for i in range(data.shape[1]):
        if np.isnan(data[0,i]):
            data[0,i] = averages["avg_" + str(i)]
    tmp = np.zeros((1, data.shape[1]))
    tmp[:] = np.nan
    data = np.concatenate((data, tmp))
    for i in range(data.shape[1]):
        if i%2 == 1:
            tmp = data[0,i]
            data[0,i] = data[1,i]
            data[1,i] = tmp
    data_normalized = BiScaler(verbose=False).fit_transform(data)
    data_filled = SoftImpute(verbose=False).fit_transform(data_normalized)
    data_filled = np.delete(data_filled, 1, 0)
    return data_filled


def clean_input(data):
    cols = data.shape[1]
    for i in range(cols):
        curr = data[:,i]
        nans = np.isnan(curr)
        if not False in nans:
            data[0,i] = averages["avg_" + str(i)]
    if data.shape[0] == 1:
        norm = np.linalg.norm(data)
        if norm == 0:
            return data
        else:
            return data / norm
    data_normalized = BiScaler(verbose=False).fit_transform(data)
    data_filled = SoftImpute(verbose=False).fit_transform(data_normalized)
    return data_filled


def count_nans(data):
    nans = np.isnan(data)
    num_count = np.count_nonzero(nans)
    nan_count = np.count_nonzero(nans==0)
    print("nans: " + str(nan_count) + "\nnums: " + str(num_count) + "\npercentage: " + str(nan_count/(nan_count+num_count)))
    return nan_count, num_count


def calculate_sepsis_score(real_labels, predicted_labels):
    t_optimal = -1
    for i in range(real_labels.shape[1]):
        if real_labels[0,i,0] == 1:
            t_optimal = i
            break
    t_sepsis = t_optimal + 6
    t_late = t_optimal + 9
    t_early = t_optimal - 6
    score = 0
    for i in range(real_labels.shape[1]):
        if t_optimal == -1:
            if predicted_labels[0,i,0] == 1:
                score -= 0.05
            else:
                continue
        else:
            if real_labels[0,i,0] == 0:
                if predicted_labels[0,i,0] == 0:
                    continue
                else:
                    score -= 0.05
            else:
                if predicted_labels[0,i,0] == 0:
                    if i >= t_late:
                        score -= 2.0
                    elif i > t_optimal:
                        score -= 0.222 * (i - t_optimal)
                    else:
                        continue
                else:
                    if i < t_early:
                        score -= 0.05
                    elif i >= t_early and i <= t_optimal:
                        score += (i - t_early) * 1.667
                    elif i > t_optimal and i <= t_late:
                        score += 1 - (t_late - i) * 0.111
                    else:
                        continue
    return score




if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 2:
        raise Exception('Include the input directory as an argument, e.g., python driver.py input .')

    input_directory = sys.argv[1]

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    # Load model.
    nans = 0
    nums = 0
    for f in files:
        # Load data.
        input_file = os.path.join(input_directory, f)
        data, labels = load_challenge_data(input_file)
        print("in file: " + f)
        tmp_nan, tmp_num = count_nans(data)
        nans += tmp_nan
        nums += tmp_num
        print("\n\n")
    print("nans: " + str(nans) + "\nnums: " + str(nums) + "\npercentage: " + str(nans/(nans+nums)))

