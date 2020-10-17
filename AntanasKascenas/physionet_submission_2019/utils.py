# Copyright (C) 2019 Canon Medical Systems Corporation. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the Physionet challenge 2019 submission project.


import pickle
import torch
from torch.nn import functional as F

from patient import Patient

def load_checkpoint(fn, model, optimiser=None, scheduler=None):
    device = torch.device("cpu") # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(fn, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimiser:
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def get_stats():
    stats = pickle.load(open("stats.pkl", "rb"))
    return stats["means"], stats["stds"]


def collate(batch):

    longest = max(batch[i][1].shape[-1] for i in range(len(batch)))

    measurements = torch.zeros(len(batch), batch[0][0].shape[0], longest)
    pus = torch.zeros(len(batch), 1, longest)
    nus = torch.zeros(len(batch), 1, longest)
    classes = torch.zeros(len(batch), 1, longest)
    one_hot_classes = torch.zeros(len(batch), 18, longest)
    pad_mask = torch.zeros(len(batch), 1, longest)

    for i, (x, pu, nu, c, h_c) in enumerate(batch):
        measurements[i, :, :x.shape[1]] = torch.tensor(x)
        pus[i, 0, :x.shape[1]] = torch.tensor(pu)
        nus[i, 0, :x.shape[1]] = torch.tensor(nu)
        classes[i, 0, :x.shape[1]] = torch.tensor(c)
        one_hot_classes[i, :, :x.shape[1]] = torch.tensor(h_c)
        pad_mask[i, 0, :x.shape[1]] = 1

    return measurements.float(), pad_mask.float(), pus.float(), nus.float(), classes.float(), one_hot_classes.float()


def TTA_predict(models, mss, pad_mask, t_add=0, t_remove=0):    
    singles, multis = [], []
    
    for model in models:
        y_pred_single, y_pred_multi = model((mss, pad_mask))
        y_pred_single *= pad_mask
        y_pred_multi *= pad_mask
        singles.append(y_pred_single)
        multis.append(y_pred_multi)

        first_pred_single = y_pred_single[..., :1]
        first_pred_multi = y_pred_multi[..., :1]


        for add in range(1, t_add+1):
            y_pred_single, y_pred_multi = model((torch.cat([mss[...,:1]]*add + [mss], dim=-1), torch.cat([pad_mask[...,:1]]*add + [pad_mask],dim=-1)))
            y_pred_single = y_pred_single[...,add:]
            y_pred_multi = y_pred_multi[...,add:]
            y_pred_single *= pad_mask
            y_pred_multi *= pad_mask
            singles.append(y_pred_single)
            multis.append(y_pred_multi)  

        for remove in range(1, min(t_remove, mss.shape[-1] - 1) + 1):
            y_pred_single, y_pred_multi = model((mss[...,remove:], pad_mask[...,remove:]))
            y_pred_single = torch.cat([first_pred_single]*remove + [y_pred_single], dim=-1)
            y_pred_multi = torch.cat([first_pred_multi]*remove +[y_pred_multi], dim=-1)
            y_pred_single *= pad_mask
            y_pred_multi *= pad_mask
            singles.append(y_pred_single)
            multis.append(y_pred_multi)

    return singles, multis

def prepare_patient(patient, device):
    batch = collate([patient.represent()])
    mss, pad_mask, _, _, _, _ = [x.to(device) for x in batch]
    return mss, pad_mask

def make_decision(y_pred_single, y_pred_multi):
    device = y_pred_single[0].device if isinstance(y_pred_single, list) else y_pred_single.device 
        
    pos = torch.tensor(Patient.utility_TP(18, 13)).view(1, 18, 1).to(device)
    neg = torch.tensor(Patient.utility_FN(18, 13)).view(1, 18, 1).to(device)

    if isinstance(y_pred_multi, list):
        ps = sum([F.softmax(x, dim=1) for x in y_pred_multi]) / len(y_pred_multi)
    else:
        ps = F.softmax(y_pred_multi, dim=1)
    
    if isinstance(y_pred_single, list):
        sigmoids = sum([x.sigmoid() for x in y_pred_single]) / len(y_pred_single)
    else:
        sigmoids = y_pred_single.sigmoid()

    E_pos = ((pos * ps) * sigmoids).sum(dim=1, keepdim=True) + ((1 - sigmoids) * (-0.05)).sum(dim=1, keepdim=True)
    E_neg = ((neg * ps) * sigmoids).sum(dim=1, keepdim=True)
    
    pred = (E_pos >= E_neg)
    decision = pred.squeeze(dim=1).squeeze(dim=0).to("cpu").numpy()[-1]
    prob = sigmoids.squeeze(dim=1).squeeze(dim=0).to("cpu").numpy()[-1]
    return decision, prob