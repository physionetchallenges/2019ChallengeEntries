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

from patient import Patient
from utils import load_checkpoint, get_stats, TTA_predict, prepare_patient, make_decision, collate
from model import ModelPassthroughSELUMultiDecision

import torch

import os


def get_sepsis_score(data, modelling_data):

    models, means, stds = modelling_data

    patient = Patient(data, None, means, stds, representation=2)

    patient_representation = patient.represent(cached=get_sepsis_score.cached if data.shape[0] != 1 else None)
    get_sepsis_score.cached = patient_representation # Store the cached representation in the function object.
    
    with torch.no_grad():
        device = next(models[0].parameters()).device
        mss, pad_mask, _, _, _, _ = collate([patient_representation])
        singles, multis = TTA_predict(models, mss[..., -32:], pad_mask[..., -32:])
        decision, prob = make_decision(singles, multis)

    return prob, decision


def load_sepsis_model():
    model_files = os.listdir("./models/")
    models = []
    for mf in model_files:
        model = ModelPassthroughSELUMultiDecision()
        _ = load_checkpoint("models/" + mf, model)
        models.append(model)
    means, stds = get_stats()

    return models, means, stds
