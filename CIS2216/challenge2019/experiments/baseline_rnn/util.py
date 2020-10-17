import os
import pandas as pd
import numpy as np


def preprocess(data_pd, stat_dict):
    # drop etco2 coz it's allways NAN.
    print('drop...')
    data_pd.drop(['EtCO2'], axis=1, inplace=True)

    for name, v in stat_dict.items():
        if 'up_limit' not in v.keys():
            # 'Unit1' has no up_limit value
            continue

        down_limit = v['down_limit']
        up_limit = v['up_limit']

        def apply_func(x):
            if x > up_limit:
                return up_limit
            if x < down_limit:
                return down_limit
            return x

        data_pd[name] = data_pd[name].apply(apply_func)

        # 2. numerical
        # fill NAN with mean value
        data_pd[name].fillna(v['mean'], inplace=True)

        # normalize
        data_pd[name] = data_pd[name].apply(lambda x: (x - v['mean']) / v['std'])

    # fill nan
    # 1. categorical unit1 and unit2
    data_pd.drop(['Unit2'], axis=1, inplace=True)
    data_pd['Unit1'].fillna(value=stat_dict['Unit1']['mode'], inplace=True)

    # dummy categorical feat
    data_pd = pd.concat([data_pd, pd.Series(data=np.array(data_pd['Unit1'].values == 0, np.float), name='Unit_0.0'),
                         pd.Series(data=np.array(data_pd['Unit1'].values == 1, np.float), name='Unit_1.0')], axis=1)
    data_pd = pd.concat([data_pd, pd.Series(data=np.array(data_pd['Gender'].values == 0, np.float), name='Gender_0'),
                         pd.Series(data=np.array(data_pd['Gender'].values == 1, np.float), name='Gender_1')], axis=1)
    data_pd.drop(['Unit1', 'Gender'], axis=1, inplace=True)

    # swap column order
    if 'SepsisLabel' in data_pd.columns:
        data_pd = pd.concat([data_pd.drop(['SepsisLabel'], axis=1), data_pd['SepsisLabel']], axis=1)  # 调整列顺序
    return data_pd


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
