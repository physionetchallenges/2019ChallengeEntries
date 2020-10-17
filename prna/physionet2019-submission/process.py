import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import collections
import copy


attributes = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', \
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', \
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', \
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', \
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', \
    'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

### Transformations
feat_transform = collections.OrderedDict()
feat_transform['BaseExcess'] = lambda x: np.log(101 - x)
feat_transform['O2Sat'] = lambda x: np.log(101-x)
featname_logTransform = ['Alkalinephos', 'AST', 'Bilirubin_direct', \
    'Bilirubin_total', 'BUN', 'Calcium', 'Creatinine', 'DBP', 'EtCO2', \
    'Fibrinogen', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Platelets', \
    'Potassium', 'PTT', 'TroponinI', 'WBC']
featname_set1 = featname_logTransform
featname_set2 = ['BaseExcess', 'O2Sat']

def remove_outlier(df_in, feat_lb_ub):
    """Given an input data frame, remove feature values outside the given range
    """
    df = copy.deepcopy(df_in)
    for feat, lb_ub in feat_lb_ub.items():
        lb, ub = lb_ub
        df.loc[df[feat]<lb, feat] = lb
        df.loc[df[feat]>ub, feat] = ub
    return df    

def logTransform(df_in, featname_set1=featname_set1, \
    featname_set2=featname_set2):
    df = copy.deepcopy(df_in)
    df[featname_set1] = np.log(df[featname_set1] + 1)
    df[featname_set2] = np.log(101 - df[featname_set2])
    return df

### Process data

def parse_data(x):
    x = x.to_dict()

    values = []

    for attr in attributes:
        if attr in x:
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values

def parse_delta(masks, dir_):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []
    height, width = masks.shape
    for h in range(height):
        if h == 0:
            deltas.append(np.ones(width))
        else:
            deltas.append(np.ones(width) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)
    
    # only used in GRU-D
    #forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values

    rec = {}

    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = []
    rec['deltas'] = deltas.tolist()

    return rec

def parse_dataMatrix(data, do_eliminate, mean, std, log_transform=False):
    if log_transform:
        data = logTransform(remove_outlier(data, feat_lb_ub={'FiO2':(0.21,1)}))
    # sz = len(data)

    # evals = []
    # for h in range(sz):
    #     evals.append(parse_data(data.iloc[h]))
        
    # # Normalize
    # evals = (np.array(evals) - mean) / std
    evals = (data.values - mean)/std

    # unroll
    shp = evals.shape; shp
    evals = evals.reshape(-1)
    
    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 10)
    
    # retain gt and set indices to nan
    values = evals.copy()
    if do_eliminate:
        values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals)) #xor

    # reshape back
    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    rec = {'label': {'forward' : [], 'backward' : []}}
    
    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')
    
    return rec

def parse_id(id_, t, do_eliminate, mean, std, log_transform=False):
    data = pd.read_csv(id_, sep='|')[:t]
    if log_transform:
        data = logTransform(remove_outlier(data, feat_lb_ub={'FiO2':(0.21,1)}))
    sz, lbl = len(data), data.SepsisLabel.values

    evals = []
    for h in range(sz):
        evals.append(parse_data(data.iloc[h]))
        
    # Normalize
    evals = (np.array(evals) - mean) / std
        
    # unroll
    shp = evals.shape; shp
    evals = evals.reshape(-1)
    
    # randomly eliminate 10% values as the imputation ground-truth
    indices = np.where(~np.isnan(evals))[0].tolist()
    indices = np.random.choice(indices, len(indices) // 10)
    
    # retain gt and set indices to nan
    values = evals.copy()
    if do_eliminate:
        values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals)) #xor

    # reshape back
    evals = evals.reshape(shp)
    values = values.reshape(shp)

    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    rec = {'label': {'forward' : lbl.tolist(), 'backward' : lbl[::-1].tolist()}}
    
    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward')
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward')
    
    return rec

def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], recs)))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals,
                'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    # Labels now a sequence, so has forward and backward
    labels = list(map(lambda x: x['label'], recs))
    ret_dict['labels'] = {}
    ret_dict['labels']['forward'] = torch.FloatTensor(list(map(lambda x: x['forward'], labels)))
    ret_dict['labels']['backward'] = torch.FloatTensor(list(map(lambda x: x['backward'], labels)))

    return ret_dict

def process_recursive(id_, path, do_eliminate=False, log_transform=False):
    '''
    Process patient .psv file and return recs for all consecutive (:t+1) sequences to train model
    '''
    mean = pd.read_csv(path.parent/'means'+('_log' if log_transform else '')+\
        '.csv', header=None, names=['param', 'value']).value.values
    std = pd.read_csv(path.parent/'stds'+('_log' if log_transform else '')+\
        '.csv', header=None, names=['param', 'value']).value.values
    
    num_rows = len(pd.read_csv(path/id_, sep='|'))
    
    all_seqs = []
    for t in range(num_rows):
        rec = parse_id(path/id_, t+1, do_eliminate, mean, std)
        data = collate_fn([rec])
        all_seqs.append(data)
    return all_seqs

def process(id_, path, do_eliminate=False, log_transform=False):
    '''
    Process patient .psv file and return rec for all full sequence
    '''
    mean = pd.read_csv(path.parent/'means'+('_log' if log_transform else '')+\
        '.csv', header=None, names=['param', 'value']).value.values
    std = pd.read_csv(path.parent/'stds'+('_log' if log_transform else '')+\
        '.csv', header=None, names=['param', 'value']).value.values
    num_rows = len(pd.read_csv(path/id_, sep='|'))
    rec = parse_id(path/id_, num_rows, do_eliminate, mean, std, \
        log_transform=logTransform)
    return collate_fn([rec])
