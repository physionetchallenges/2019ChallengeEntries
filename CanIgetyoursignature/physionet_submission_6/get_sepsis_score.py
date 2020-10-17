#!/usr/bin/env python
from definitions import *
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from src.submission.generate_submission.main import *


def load_sepsis_model():
    loc = ROOT_DIR + '/models/submissions/submission_6_files'
    fname1 = loc + '/clf.pickle'
    fname3 = loc + '/threshold.pickle'

    models = {
        'clf': load_pickle(fname1),
        'threshold': load_pickle(fname3)
    }
    return models

def make_frame(data, column_names):
    """ Puts in the dataframe form that can be used by the algos """
    df = pd.DataFrame(data=data, columns=column_names)

    # Add the time index
    df.index = pd.MultiIndex.from_tuples((1, time) for time in range(df.shape[0]))
    df.index.names = ['id', 'time']

    return df

def get_sepsis_score(data, models):
    
    # Open the models
    clf = models['clf']
    threshold = models['threshold']

    # Get col names
    column_names = load_pickle(MODELS_DIR + '/other/col_names.pickle')

    # Make df (drop hospital col)
    df = make_frame(data, column_names[0:40])
    # idx = df.index.get_level_values('time')[-1]

    # Now process the data
    df = basic_data_process(df)
    df = process_data_and_build(df, config['feature'], submission=True)

    # Now make final prediction
    pred = clf.predict(df)

    # test_equate(pred, idx)
    # Turn into 0, 1
    pred = (pred > threshold).astype(int)

    return pred[0], pred[0]

def test_equate(pred, idx):
    p = load_pickle(ROOT_DIR + '/models/submissions/submission_6_files/predictions.pickle').loc[9, idx]
    ppprint("WHAT THE U")
    if pred[0] != p:
        print('h')
    # assert pred[0] == p, str(pred[0]) + ':::' + str(p)
    # if pred[0] == p:
    #     ppprint('YE:S ' + str(idx))
    # else:
    #     AssertionError("FUCK")

