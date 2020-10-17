#!/usr/bin/env python

import numpy as np, os, sys
from get_sepsis_score_fast import load_sepsis_model, get_sepsis_score
from joblib import Parallel, delayed

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))

def run_predictions(i, f, model, input_directory, output_directory):
        print('    {}/~20000...'.format(i+1))

        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)

        """ Make predictions faster
        """
        if data.size != 0:
            (scores, labels) = get_sepsis_score(data, model)

        # Save results.
        output_file = os.path.join(output_directory, f)
        save_challenge_predictions(output_file, scores, labels)
        
        
if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading sepsis model...')
    model = load_sepsis_model()

    # Iterate over files.
    print('Predicting sepsis labels...')
    num_files = len(files)
    Parallel(n_jobs=2)(delayed(run_predictions)(i, f, model, input_directory, output_directory) for i, f in enumerate(files))

    print('Done.')
