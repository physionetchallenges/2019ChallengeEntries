from fastai.core import *


def get_data_fields(params:dict, cat_mode:int=1):
    cats, conts, cats_converted = [list(params[key].keys()) for key in ['cat_missing', 'cont_stats', 'cont_to_cats']]
    for l in (cats, conts, cats_converted): l.sort()

    if cat_mode == 1: conts = [c for c in conts if c not in cats_converted]
    if cat_mode > 0: cats += [f'{c}_CAT_' if cat_mode > 1 else c for c in cats_converted]
    times = params['time_fields']
    return times, conts, cats

def preprocess_data(data: pd.DataFrame, params:dict, cat_mode:int, norm:str='z'):
    interpolate_thres = 0.2 if 'interpolate_thres' not in params else params['interpolate_thres']
    
    _drop_field(data, 'EtCO2')
    data['Age'] = data.Age.apply(_cat_age)

    if cat_mode > 0:
        for field, bounds in params['cont_to_cats'].items():
            target = field if cat_mode == 1 else f'{field}_CAT_'
            data[target] = data[field].apply(partial(_cat_contigous_values, points=bounds))    
    _fill_missing(data, params, interpolate_thres=interpolate_thres)
    _norm_fields(data, params, norm=norm)
    return data

def _drop_field(data: pd.DataFrame, fields:OptStrList=None):
    if fields is None: return data
    data.drop(columns=fields, inplace=True)
    return data

def _fill_missing(data: pd.DataFrame, params:dict, interpolate_thres:float=0.2):
    for col in data.columns:
        na_mask = data[col].isna()
        missed_count = len(data.loc[na_mask, col])
        if missed_count == 0: continue
        if col in params['cat_missing']: data.loc[na_mask, col] = params['cat_missing'][col]
        elif col in params['cont_missing']: 
            if missed_count<len(data)*interpolate_thres:
                data.loc[data.index, col] = data.loc[data.index, col].interpolate(kind='linear', limit_direction='both')
            else: data.loc[na_mask, col] = params['cont_missing'][col]
        else: raise AttributeError(f'Unkown field {col}')
    return data

def _norm_fields(data: pd.DataFrame, params:dict, eps=1e-9, norm='z'):
    _, _, cats = get_data_fields(params)
    for col in data:
        if col in cats: continue
        if col not in params['cont_stats']: continue
        (mean, std, min_v, max_v), value = params['cont_stats'][col], data[col]
        if norm == 'r': data[col] = (value - min_v)/ (max_v - min_v + eps)
        elif norm == 'z': data[col] = (value - mean)/(std + eps)
        elif norm == 'zr': 
            min_v, max_v = mean - 4*std, mean + 4*std
            data.loc[value < min_v, col] = min_v
            data.loc[value > max_v, col] = max_v
            data[col] = ((data[col] - min_v)/(max_v - min_v + eps)) - 0.5
    return data

def _cat_contigous_values(value, points=[], equal_len=True):
    assert equal_len, "We currently just support the equal length"
    if not isinstance(value, Number) or np.isnan(value): return len(points) + 1
    if value <= points[0]: return 0
    elif value >= points[-1]: return len(points)
    else: return int(np.ceil((value - points[0])/(points[1] - points[0]))) 

def _cat_age(value): 
    value = int(value)
    if value <= 18: return 0
    if 18 < value <= 25: return 1
    if 25 < value <= 35: return 2
    if 35 < value <= 45: return 3
    if 45 < value <= 55: return 4
    if 55 < value <= 65: return 5
    if 65 < value <= 70: return 6
    if 70 < value <= 75: return 7
    if 75 < value <= 80: return 8
    return 9
