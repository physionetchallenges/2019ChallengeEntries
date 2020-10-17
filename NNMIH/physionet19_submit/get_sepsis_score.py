#!/usr/bin/env python

import os, torch, pickle
import torch.nn.functional as F
import numpy as np
from glob import glob

from modules.preprocessor import manual_processor_v4 as processor

def do_imputation(data):
    data[np.isnan(data)] = 0
    return data

def do_scaling(data, scaler):
    data = scaler.transform(data)
    return data

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else: device = "cpu"
    return device

def load_model(net_type, argsfile, paramfile):
    
    if net_type == "GRUv10":
        from modules.module_nets_gruv10 import GRUv10 as Model
    elif net_type == "GRUv31":
        from modules.module_nets_gruv31 import GRUv31 as Model

    args = pickle.load(open(argsfile, "rb"))
    args.device = get_device()

    model = Model(args, base_list=None)
    model.load_state_dict(torch.load(paramfile, map_location='cpu'))
    return model.to(args.device)

def predict_by_model(model, data, M, mask):
    
    with torch.no_grad():
        model.eval()
        pred = model(data, M, mask)
        return pred
    
def get_sepsis_score(data, model):
    models, scaler = model
    models1, models2 = models
    
    thres = 0.5
    scores = []
    
    _data = data.copy()
    M = (~np.isnan(_data))*1.
    M = torch.from_numpy(M).unsqueeze(1).float()
    _data = processor(_data)
    _data = do_scaling(_data, scaler)
    _data = do_imputation(_data)
    _data = torch.from_numpy(_data).unsqueeze(1)
    _data = _data.float()
    mask = torch.ones_like(_data[:, :, 0])

    device = get_device()
    _data, mask, M = _data.to(device), mask.to(device), M.to(device)

    out1 = []
    for m1 in models1:
        pred1 = predict_by_model(m1, _data, M, mask)
        out1.append(pred1)
    out1 = torch.cat(out1, dim=-1)

    for i, m2 in enumerate(models2):
        with torch.no_grad():
            m2.eval()
            cat_d = torch.cat([out1, _data], dim=-1)
            _m = torch.ones_like(out1).to(M.device)
            cat_m = torch.cat([_m, M], dim=-1)

            pred2 = m2(cat_d, cat_m, mask)
            pred2 = F.softmax(pred2, dim=-1)[:, :, 1]
            pred2 = pred2.detach().cpu().numpy()
            scores = pred2.flatten()
            
    score = scores[-1]
    label = 1.*(score > 0.5)
    return score, label

def load_sepsis_model():
    scalerfile = "./modules/module_scaler.pkl"
    args_base = "./modules/models/m{:02d}_args.pkl"
    prms_base = "./modules/models/m{:02d}_params.pth"
    net_types1 = ["GRUv10", "GRUv10", "GRUv10", "GRUv10", 
                  "GRUv10", "GRUv10"]
    net_types2 = ["GRUv31"]

    models1, models2 = [], []
    for i, net_type in enumerate(net_types1):
        argsfile = args_base.format(i+1)
        paramfile = prms_base.format(i+1)
        model = load_model(net_type, argsfile, paramfile)
        models1.append(model)

    for j, net_type in enumerate(net_types2):
        argsfile = args_base.format(i+j+2)
        paramfile = prms_base.format(i+j+2)
        model = load_model(net_type, argsfile, paramfile)
        models2.append(model)

    models = [models1, models2]
    scaler = pickle.load(open(scalerfile, "rb"))
    return models, scaler
