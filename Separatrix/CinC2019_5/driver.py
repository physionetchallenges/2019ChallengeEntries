#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Morteza Zabihi (morteza.zabihi@gmail.com) 
(June 2019)
# The code is borrowed from: https://physionet.org/challenge/2019/
=============================================================================== 
This code is released under the GNU General Public License.
You should have received a comodelspy of the GNU General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

By accessing the code through Physionet webpage and/or by installing, copying, or otherwise
using this software, you agree to be bounded by the terms of GNU General Public License.
If you do not agree, do not install copy or use the software.

We make no representation about the suitability of this licensed deliverable for any purppose.
It is provided "as is" without express or implied warranty of any kind.
Any use of the licensed deliverables must include the above disclaimer.
For more detailed information please refer to "readme" file.
===============================================================================
"""
import numpy as np, os, sys
from get_sepsis_score import load_sepsis_model, get_sepsis_score

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
    model = load_sepsis_model()

    # Iterate over files.
    for f in files:
        # Load data.
        input_file = os.path.join(input_directory, f)
        data = load_challenge_data(input_file)

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