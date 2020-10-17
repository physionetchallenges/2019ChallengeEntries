#!/usr/bin/env python

import numpy as np
import pandas as pd
import torch
from model import RNNModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocessing(data):
    """
    Data preprocessing
    - drop variable
    - mask
    - interpolation
    - scaling
    - to torch
    :param data:
    :return:
    """
    # numpy to pandas
    variable_names = np.load('saved_items/variable_names.npy', allow_pickle=True)
    data = pd.DataFrame(data, columns=variable_names)

    # ----- FEATURE SELECTION --------------------
    selected_variables = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
                          'HCO3', 'FiO2', 'pH', 'Creatinine', 'Glucose', 'Hgb', 'WBC', 'Platelets',
                          'Age', 'Gender', 'ICULOS']
    data = data.loc[:, selected_variables]

    # ----- MASK --------------------
    mask = data.where(data.isna(), 1).fillna(0).astype(int)

    # ----- SCALING --------------------
    mean_x = pd.read_csv('saved_items/mean_value.psv', sep='|').transpose()
    std_x = pd.read_csv('saved_items/std_value.psv', sep='|').transpose()
    reference_values = pd.read_csv('saved_items/reference_values.psv', sep='|').transpose()
    std_ref = pd.read_csv('saved_items/std_ref.psv', sep='|').transpose()

    # SCALE WITH REFERENCE
    columns_not_to_scale = ['Age', 'Gender', 'ICULOS']
    ref_matrix = pd.concat([reference_values] * data.shape[0], axis=1, ignore_index=True).transpose()
    ref_matrix = ref_matrix.loc[:, data.columns.difference(columns_not_to_scale)]
    std_matrix = pd.concat([std_ref] * data.shape[0], axis=1, ignore_index=True).transpose()
    std_matrix = std_matrix.loc[:, data.columns.difference(columns_not_to_scale)]
    data.loc[:, ref_matrix.columns.values] = (data.loc[:, ref_matrix.columns.values] - ref_matrix) / std_matrix

    # STANDARDIZE AGE AND ICULOS BY MEAN
    mean_matrix = pd.concat([mean_x] * data.shape[0], axis=1, ignore_index=True).transpose()
    std_matrix = pd.concat([std_x] * data.shape[0], axis=1, ignore_index=True).transpose()
    columns_standardize = ['Age', 'ICULOS']
    mean_matrix = mean_matrix.loc[:, columns_standardize]
    std_matrix = std_matrix.loc[:, columns_standardize]
    data.loc[:, columns_standardize] = (data.loc[:, columns_standardize] - mean_matrix) / std_matrix

    # ----- INTERPOLATE --------------------
    # Columns full of NaNs -> interpolate with reference of variable
    nan_columns = data.isna().all()
    idx_nan_columns = nan_columns[nan_columns].index.values
    ref_matrix = pd.concat([reference_values] * data.shape[0], axis=1, ignore_index=True).transpose()
    data[idx_nan_columns] = ref_matrix[idx_nan_columns]

    # forward fill
    data = data.fillna(method='ffill')
    data = data.fillna(method='backfill')
    # if remaining NaN
    data = data.fillna(0.)

    # ----- TO TORCH --------------------
    data = np.expand_dims(data.to_numpy(), axis=2)
    mask = np.expand_dims(mask.to_numpy(), axis=2)
    data_aggregation = np.concatenate((data, mask), axis=2)

    data_aggregation = torch.FloatTensor(data_aggregation)

    return data_aggregation


def get_sepsis_score(data, model):
    # ----- PRE-PROCESSING --------------------
    data = preprocessing(data)

    model.eval()
    with torch.no_grad():
        # input ->(1 x sequence_length x variable x 2)
        data = data.unsqueeze_(dim=0).to(device)
        sequence_length = torch.as_tensor(data.size(0), dtype=torch.int64, device=device).unsqueeze_(dim=0)

        # ----- MAKE PREDICTION --------------------
        dataset = [data, sequence_length]
        outputs = model(dataset)

        _, predicted = torch.max(outputs, 1)
        predicted_sepsis_prob = outputs[:, 1]

        score = predicted_sepsis_prob.squeeze_().cpu().numpy()
        label = predicted.squeeze_().cpu().numpy()

    return score, label


def load_sepsis_model():
    """
    Load model
    :return:
    """
    model = RNNModel(18, 100, 2, nb_layer=2, add_mask=True)
    model.load_state_dict(torch.load('saved_items/model_state_dict.pth', map_location='cpu'))

    return model
