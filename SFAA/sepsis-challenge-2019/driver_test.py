#!/usr/bin/env python

import numpy as np, os, sys
import shutil
from  compute_scores_2019 import compute_scores_2019
from get_sepsis_score import load_sepsis_model, get_sepsis_score

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # # Ignore SepsisLabel column if present.
    # if column_names[-1] == 'SepsisLabel':
    #     column_names = column_names[:-1]
    #     data = data[:, :-1]

    return data
def save_labels(file, labels):
    with open(file, 'w') as f:
        f.write('SepsisLabel\n')
        for l in labels:
            f.write('%d\n' % l)
def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))

if __name__ == '__main__':
    # Parse arguments.
    # if len(sys.argv) != 3:
    #     raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = '../training_setB' #sys.argv[1]
    output_directory = 'predictions_directory_ch_x2' #sys.argv[2]
    # remove them to challenge
    labels_directory = 'labels_directory_ch_x2'

    if (os.path.exists(labels_directory) == True):
        shutil.rmtree(labels_directory)
    os.makedirs(labels_directory)

    if (os.path.exists(output_directory) == True):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)


    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # remove them to challenge!
    files = sorted(files)
    files = files[-2000:]

    # Load model.
    print('Loading sepsis model...')
    model = load_sepsis_model()

    # Iterate over files.
    print('Predicting sepsis labels...')
    num_files = len(files)
    for i, f in enumerate(files):
        print('    {}/{}...'.format(i+1, num_files))

        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)

        # remove them to challenge!
        # read label
        label_t = data[:, -1]
        data = data[:,:-1]
        # Make predictions.
        num_rows = len(data)
        scores = np.zeros(num_rows)
        labels = np.zeros(num_rows)
        for t in range(num_rows):
            current_data = data[:t+1]
            current_score, current_label = get_sepsis_score(current_data, model)
            scores[t] = current_score
            labels[t] = current_label

        # Save results.
        output_file = os.path.join(output_directory, f)
        save_challenge_predictions(output_file, scores, labels)

        # remove them to challenge!
        output_file_l = os.path.join(labels_directory, f)
        save_labels(output_file_l, label_t)
    # remove them for challenge!
    auroc, auprc, accuracy, f_measure, utility = compute_scores_2019(labels_directory,output_directory)

    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Utility:\n{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}'.format(auroc, auprc,
                                                                                                         accuracy,
                                                                                                         f_measure,
                                                                                                         utility)
    print(output_string)

    print('Done.')
