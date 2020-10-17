#!/usr/bin/env python

import numpy as np
from model import RitsModel, TCN
import torch
from process import parse_dataMatrix, collate_fn, attributes
from utils_jr import to_var
import pandas as pd


flag_useCuda = False
device = torch.device('cpu')
if flag_useCuda:
    torch.cuda.set_device(0)
    device = torch.device('cuda')

def get_sepsis_score(data, model):
    # normalize isMeasured variables
    # only take the last 10 measurements (due to the receptive field of TCN)
    idx_start = max(0, data.shape[0]-10)
    x_isMeasured = ~np.isnan(data[idx_start:,range(0,34)])
    x_isMeasured_norm = torch.from_numpy((x_isMeasured-model['isMeasured_mean'])/\
        model['isMeasured_std']).float().to(device)
    # run RITS imputation: apply log-transform before running RITS
    df = pd.DataFrame(data, columns=attributes)
    rec = parse_dataMatrix(df, False, model['rits_mean'], model['rits_std'], \
        log_transform=True)
    seq = collate_fn([rec])
    # get the number of models
    n_model = len(model['ritsModel_list'])
    score = 0.
    with torch.no_grad():
        seq_var = to_var(seq, device=device)
        for idx in range(n_model):
            rits_model = model['ritsModel_list'][idx]
            tcn_model = model['tcnModel_list'][idx]
            tcn_threshold = model['tcnThreshold_list'][idx]
            # apply RITS imputation
            _, ret = rits_model.run_on_batch([seq_var], None)
            imputation = ret[0]['imputations']
            # concatenate imputed data (of last 10 measurements) and isMeasured
            x_tensor = torch.cat((imputation[0][idx_start:], x_isMeasured_norm), 1)
            # get the prediction of the last element
            pred_tcn = tcn_model.forward(x_tensor)[-1].item()
            score += (pred_tcn>tcn_threshold)
    score = score*1./n_model
    label = int(score>(1./3+1e-3))
    return score, label

def load_sepsis_model():
    # the list of rits models, tcn models, tcn thresholds
    ritsModel_list, tcnModel_list, tcnThreshold_list = [], [], []
    # load multiple models iteratively
    for testFold in [9, 3, 7]:
        # load RITS models
        rits_model = RitsModel(40, 64, 0.5, device=device)
        path_rits = "./trained_models/RITSLog_testFold"+str(testFold)+".pkl"
        if flag_useCuda:
            rits_model.load_state_dict(torch.load(path_rits)['model_state_dict'])
        else:
            rits_model.load_state_dict(torch.load(path_rits, map_location=\
                lambda storage, loc: storage)['model_state_dict'])
        rits_model = rits_model.eval()
        if flag_useCuda:
            rits_model = rits_model.cuda()
        ritsModel_list.append(rits_model)

        # load TCN models and thresholds
        tcn_model = TCN(74, 1, [100], kernel_size=5, dropout=0.25)
        path_tcn = "./trained_models/TCNmodel_testFold"+str(testFold)+".pt"
        if flag_useCuda:
            tcn_model.load_state_dict(torch.load(path_tcn))
        else:
            tcn_model.load_state_dict(torch.load(path_tcn, map_location=\
                lambda storage, loc: storage))           
        tcn_model = tcn_model.eval()
        if flag_useCuda:
            tcn_model = tcn_model.cuda()
        tcnModel_list.append(tcn_model)

        tcn_threshold = np.load("./trained_models/TCNthreshold_testFold"+\
            str(testFold)+".npy")[0]
        tcnThreshold_list.append(tcn_threshold)

    # load normalization mean, std for 34 vital+lab imputed by rits
    rits_mean = pd.read_csv("./trained_models/means_log.csv", header=None, \
        names=['param','value']).value.values
    rits_std = pd.read_csv("./trained_models/stds_log.csv", header=None, \
        names=['param','value']).value.values

    # load normalization mean, std for binary variables used by TCN
    isMeasured_mean = np.loadtxt("./trained_models/isMeasured_mean.csv")
    isMeasured_std = np.loadtxt("./trained_models/isMeasured_std.csv")

    return {'ritsModel_list':ritsModel_list, 'tcnModel_list':tcnModel_list, \
        'tcnThreshold_list':tcnThreshold_list, 'rits_mean':rits_mean, \
        'rits_std':rits_std, 'isMeasured_mean':isMeasured_mean, \
        'isMeasured_std':isMeasured_std}
