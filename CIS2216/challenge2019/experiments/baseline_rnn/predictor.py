import pickle
import sys
import pandas as pd

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from util import compute_prediction_utility, preprocess
from model import MyRNN


class PreditorWrapper(object):
    def __init__(self, model, model_path='s', therhold=0.3):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # load model
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)
        model.to(device)

        self.model = model
        self.therhold = therhold
        self.device = device

    def predict(self, input):
        input = torch.FloatTensor(input).to(self.device).unsqueeze(dim=0)
        logits = self.model(input)
        prob = F.softmax(logits, dim=-1)[0, :, 1]
        pred = torch.where(prob > self.therhold, torch.ones_like(prob), torch.zeros_like(prob))

        prob = prob.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()

        return prob, pred


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: %s input[.psv]' % sys.argv[0])

    record_name = sys.argv[1]
    if record_name.endswith('.psv'):
        record_name = record_name[:-4]

    # read input data
    input_file = record_name + '.psv'
    df = pd.read_table(input_file, sep='|')
    if 'SepsisLabel' in df.columns:
        df = df.drop(['SepsisLabel'], axis=1)

    # read stat
    stat_dict = pickle.load(open('./binary_files/normalized_stat_dict.pkl', 'rb'))

    # fill in missing value
    df = preprocess(df, stat_dict)

    # read model
    model = PreditorWrapper(MyRNN(df.shape[1]), './binary_files/model.th')

    # generate prediction
    scores, labels = model.predict(df.values)

    # write predictions to output file
    output_file = record_name + '.out'
    with open(output_file, 'w') as f:
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))
