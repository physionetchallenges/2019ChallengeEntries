import os
import pandas as pd
import numpy as np
import pickle
import dill
from collections import Counter


def multi_vote(models_dict, values):
    all_pred = []
    for k, v in models_dict.items():
        clf, thresh, _ = v
        tmp_pred_proba = clf.predict_proba(values)[:, 1]
        tmp_pred = np.array(tmp_pred_proba > thresh, dtype=np.int32)
        all_pred.append(tmp_pred)
    all_pred = np.array(all_pred)
    vote_pred_prob = np.mean(all_pred, axis=0)
    vote_pred = np.array(vote_pred_prob > 0.5, dtype=np.int32)

    return vote_pred_prob, vote_pred


def sample_model(values):
    x_mean = np.array([
        83.8996, 97.0520, 36.8055, 126.2240, 86.2907,
        66.2070, 18.7280, 33.7373, -3.1923, 22.5352,
        0.4597, 7.3889, 39.5049, 96.8883, 103.4265,
        22.4952, 87.5214, 7.7210, 106.1982, 1.5961,
        0.6943, 131.5327, 2.0262, 2.0509, 3.5130,
        4.0541, 1.3423, 5.2734, 32.1134, 10.5383,
        38.9974, 10.5585, 286.5404, 198.6777])
    x_std = np.array([
        17.6494, 3.0163, 0.6895, 24.2988, 16.6459,
        14.0771, 4.7035, 11.0158, 3.7845, 3.1567,
        6.2684, 0.0710, 9.1087, 3.3971, 430.3638,
        19.0690, 81.7152, 2.3992, 4.9761, 2.0648,
        1.9926, 45.4816, 1.6008, 0.3793, 1.3092,
        0.5844, 2.5511, 20.4142, 6.4362, 2.2302,
        29.8928, 7.0606, 137.3886, 96.8997])
    c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
    c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

    x = values[:, 0:34]
    c = values[:, 34:40]
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    c_norm = np.nan_to_num((c - c_mean) / c_std)

    beta = np.array([
        0.1806, 0.0249, 0.2120, -0.0495, 0.0084,
        -0.0980, 0.0774, -0.0350, -0.0948, 0.1169,
        0.7476, 0.0323, 0.0305, -0.0251, 0.0330,
        0.1424, 0.0324, -0.1450, -0.0594, 0.0085,
        -0.0501, 0.0265, 0.0794, -0.0107, 0.0225,
        0.0040, 0.0799, -0.0287, 0.0531, -0.0728,
        0.0243, 0.1017, 0.0662, -0.0074, 0.0281,
        0.0078, 0.0593, -0.2046, -0.0167, 0.1239])
    rho = 7.8521
    nu = 1.0389

    xstar = np.concatenate((x_norm, c_norm), axis=1)
    exp_bx = np.exp(np.matmul(xstar, beta))
    l_exp_bx = pow(4 / rho, nu) * exp_bx

    scores = 1 - np.exp(-l_exp_bx)
    labels = (scores > 0.45)
    return labels


def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2,
                               u_fp=-0.05, u_tn=0):
    # Check inputs for errors.
    if len(predictions) != len(labels):
        raise Exception('Numbers of predictions and labels must be the same.')

    n = len(labels)
    for i in range(n):
        if not labels[i] in (0, 1):
            raise Exception('Labels must satisfy label == 0 or label == 1.')

    for i in range(n):
        if not predictions[i] in (0, 1):
            raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

    if dt_early >= dt_optimal:
        raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

    if dt_optimal >= dt_late:
        raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if any(labels):
        is_septic = True
        t_sepsis = min(i for i, label in enumerate(labels) if label)
    else:
        is_septic = False
        t_sepsis = float('inf')

    # Define slopes and intercept points for affine utility functions of the
    # form u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u), u


def make_data():
    res = {'pid': [], 'data': [], 'final_label': []}

    path1 = 'data/training_setA/'
    for fname in os.listdir(path1):
        df = pd.read_table('{0}{1}'.format(path1, fname), sep='|')
        res['pid'].append(fname.split('.')[0])
        res['data'].append(df)
        if np.sum(df.loc[:, 'SepsisLabel'].values) > 0:
            res['final_label'].append(1)
        else:
            res['final_label'].append(0)

    path2 = 'data/training_setB/'
    for fname in os.listdir(path2):
        df = pd.read_table('{0}{1}'.format(path2, fname), sep='|')
        res['pid'].append(fname.split('.')[0])
        res['data'].append(df)
        if np.sum(df.loc[:, 'SepsisLabel'].values) > 0:
            res['final_label'].append(1)
        else:
            res['final_label'].append(0)

    print(Counter(res['final_label']))
            
    with open('data/data.pkl', 'wb') as fout:
        dill.dump(res, fout)

