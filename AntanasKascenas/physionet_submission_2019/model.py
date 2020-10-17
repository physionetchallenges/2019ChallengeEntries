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


import torch
from torch import nn
from torch.nn import functional as F

from patient import Patient
from utils import collate


class PassThrough(nn.Module):
    """
    Passthrough layer used for adjusting weights/biases for initialisation and applying padding masks
    """

    def __init__(self, weights, biases):
        super(PassThrough, self).__init__()
        self.weights = weights
        self.biases = biases

    def forward(self, x, pad_mask):
        return x * pad_mask


class ModelPassthroughSELUMultiDecision(nn.Module):
    def __init__(self, n_vars=34, embedding_dim=10, n_features=16):
        super(ModelPassthroughSELUMultiDecision, self).__init__()

        self.disable_dropout()
        
        self.n_vars = n_vars
        self.embedding_dim = embedding_dim
        self.n_features = n_features

        self.emb_conv1 = nn.Conv1d(self.n_features * self.n_vars, self.embedding_dim * self.n_vars * 2, 1,
                                   groups=self.n_vars)
        self.emb_selu1 = nn.SELU()
        self.emb_passthrough1 = PassThrough(weights=[self.emb_conv1.weight], biases=[self.emb_conv1.bias])
        self.emb_conv2 = nn.Conv1d(self.embedding_dim * self.n_vars * 2, self.embedding_dim * self.n_vars * 2, 1,
                                   groups=self.n_vars)
        self.emb_selu2 = nn.SELU()
        self.emb_passthrough2 = PassThrough(weights=[self.emb_conv2.weight], biases=[self.emb_conv2.bias])

        self.emb_conv3 = nn.Conv1d(self.embedding_dim * self.n_vars * 2, self.embedding_dim * self.n_vars, 1,
                                   groups=self.n_vars)
        self.emb_selu3 = nn.SELU()
        self.emb_passthrough3 = PassThrough(weights=[self.emb_conv3.weight], biases=[self.emb_conv3.bias])

        self.patient_conv1 = nn.Conv1d(self.n_vars * self.embedding_dim, 256, 1)
        self.patient_selu1 = nn.SELU()
        self.patient_passthrough1 = PassThrough(weights=[self.patient_conv1.weight], biases=[self.patient_conv1.bias])
        self.patient_conv2 = nn.Conv1d(256, 256, 1)
        self.patient_selu2 = nn.SELU()
        self.patient_passthrough2 = PassThrough(weights=[self.patient_conv2.weight], biases=[self.patient_conv2.bias])

        self.pad1 = nn.ConstantPad1d((1, 0), 0)
        self.pad2 = nn.ConstantPad1d((2, 0), 0)
        self.pad4 = nn.ConstantPad1d((4, 0), 0)
        self.pad8 = nn.ConstantPad1d((8, 0), 0)

        
        self.conv1 = nn.Conv1d(256, 128, 2)
        self.selu1 = nn.SELU()
        self.passthrough1 = PassThrough(weights=[self.conv1.weight], biases=[self.conv1.bias])
        self.conv2 = nn.Conv1d(128, 128, 2)
        self.conv_r1 = nn.Conv1d(256, 128, 1)
        self.selu_r1 = nn.SELU()
        self.passthrough_r1 = PassThrough(weights=[self.conv_r1.weight, self.conv2.weight], biases=[self.conv2.bias])

        self.conv3 = nn.Conv1d(128, 128, 2, dilation=2)
        self.selu3 = nn.SELU()
        self.passthrough3 = PassThrough(weights=[self.conv3.weight], biases=[self.conv3.bias])
        self.conv4 = nn.Conv1d(128, 128, 2, dilation=2)
        self.conv_r2 = nn.Conv1d(128, 128, 1)
        self.selu_r2 = nn.SELU()
        self.passthrough_r2 = PassThrough(weights=[self.conv_r2.weight, self.conv4.weight], biases=[self.conv4.bias])

        self.conv5 = nn.Conv1d(128, 64, 2, dilation=4)
        self.selu5 = nn.SELU()
        self.passthrough5 = PassThrough(weights=[self.conv5.weight], biases=[self.conv5.bias])
        self.conv6 = nn.Conv1d(64, 64, 2, dilation=4)
        self.conv_r3 = nn.Conv1d(128, 64, 1)
        self.selu_r3 = nn.SELU()
        self.passthrough_r3 = PassThrough(weights=[self.conv_r3.weight, self.conv6.weight], biases=[self.conv6.bias])
        
        self.conv7 = nn.Conv1d(64, 64, 2, dilation=8)
        self.selu7 = nn.SELU()
        self.passthrough7 = PassThrough(weights=[self.conv7.weight], biases=[self.conv7.bias])
        self.conv8 = nn.Conv1d(64, 64, 2, dilation=8)
        self.conv_r4 = nn.Conv1d(64, 64, 1)
        self.selu_r4 = nn.SELU()
        self.passthrough_r4 = PassThrough(weights=[self.conv_r4.weight, self.conv8.weight], biases=[self.conv8.bias])

        #self.fdropout = nn.AlphaDropout(p=0.2)
        #self.fconv = nn.Conv1d(64, 18, 1)
        self.fconv = nn.Conv1d(64, 18, 1)
        #self.fconv2 = nn.Conv1d(64, 1, 1)
        self.fconv2 = nn.Conv1d(64, 1, 1)
        

    def find_modules(self, m, cond):
        if cond(m): return [m]
        return sum([self.find_modules(o,cond) for o in m.children()], [])
    
    
    def check_stats(self, dataloader, device):
    
        ms, pad_mask, pu, nu, classes, one_hot_classes = next(dataloader.__iter__())
        ms = ms.to(device)
        pad_mask = pad_mask.to(device)

        def save_stats(self, input, output):
            eps = 0.001
            output = output * pad_mask
            c = pad_mask.sum()*output.shape[1]
            m = output.sum()/c
            v = (output**2).sum()/c - m**2
            self._saved_mean = m.item()
            self._saved_std = (v + eps).sqrt().item()

        mods = self.find_modules(model, lambda x: isinstance(x, PassThrough))
        print(mods)
        hooks = []
        for m in mods:
            hooks.append(m.register_forward_hook(save_stats))

        with torch.no_grad():
            self((ms, pad_mask))

        print("Batch statistics:")
        for m in mods:
            print(m._saved_mean, m._saved_std)

        for h in hooks:
            h.remove()

        return mods
        
    def enable_weight_norm(self):
        for m in self.find_modules(self, lambda x: isinstance(x, nn.Conv1d)):
            weight_norm(m)
        
    def lsuv_init(self, dataloader, device, max_it=40):

        ms, pad_mask, pu, nu, classes, one_hot_classes = next(dataloader.__iter__())
        ms = ms.to(device)
        pad_mask = pad_mask.to(device)

        def save_stats(self, input, output):
            output = output * pad_mask
            eps = 0.001
            c = pad_mask.sum() * output.shape[1]
            m = output.sum() / c
            v = (output ** 2).sum() / c - m ** 2
            self._saved_mean = m.item()
            self._saved_std = (v + eps).sqrt().item()

        mods = self.find_modules(self, lambda x: isinstance(x, PassThrough))
        print(mods)
        hooks = []
        for m in mods:
            hooks.append(m.register_forward_hook(save_stats))

        self((ms, pad_mask))

        print("Batch statistics in the beginning:")
        for m in mods:
            print(m._saved_mean, m._saved_std)

        with torch.no_grad():
            for m in mods:
                it = 0

                while self((ms, pad_mask)) is not None and (
                        (abs(m._saved_mean) > 1e-3 or abs(m._saved_std - 1) > 1e-3) and it < max_it):
                    # print(it)
                    for bias in m.biases:
                        bias.sub_(m._saved_mean)
                    for weight in m.weights:
                        weight.data.div_(m._saved_std)
                    it += 1

        print("Batch statistics after init:")
        for m in mods:
            print(m._saved_mean, m._saved_std)

        for h in hooks:
            h.remove()

    def embed(self, ms, pad_mask):
        
        res = self.emb_passthrough1(self.emb_selu1(self.emb_conv1(ms)), pad_mask)
        res = self.emb_passthrough2(self.emb_selu2(self.emb_conv2(res)), pad_mask)
        res = self.emb_passthrough3(self.emb_selu3(self.emb_conv3(res)), pad_mask)
        return res

    def patient_embed(self, emb, pad_mask):
        res = self.patient_passthrough1(self.patient_selu1(self.patient_conv1(emb)), pad_mask)
        res = self.patient_passthrough2(self.patient_selu2(self.patient_conv2(res)), pad_mask)

        return res

    def enable_dropout(self):
        self.dropout_enabled = True
    
    def disable_dropout(self):
        self.dropout_enabled = False
    
    def classify(self, emb, pad_mask):
        x = self.passthrough1(self.selu1(self.conv1(self.pad1(emb))), pad_mask)
        x1 = self.passthrough_r1(self.selu_r1(self.conv_r1(emb) + self.conv2(self.pad1(x))), pad_mask)
        x = self.passthrough3(self.selu3(self.conv3(self.pad2(x1))), pad_mask)
        x2 = self.passthrough_r2(self.selu_r2(self.conv_r2(x1) + self.conv4(self.pad2(x))), pad_mask)
        x = self.passthrough5(self.selu5(self.conv5(self.pad4(x2))), pad_mask)
        x3 = self.passthrough_r3(self.selu_r3(self.conv_r3(x2) + self.conv6(self.pad4(x))), pad_mask)
        x = self.passthrough7(self.selu7(self.conv7(self.pad8(x3))), pad_mask)
        x4 = self.passthrough_r4(self.selu_r4(self.conv_r4(x3) + self.conv8(self.pad8(x))), pad_mask)
        x =  x4  # self.fdropout(x)
        
        #att_logits = self.att_logits(x)
        #cumsum = torch.cumsum(att_logits, dim=-1)
        
        #attended_x = 
        #avg = torch.cumsum(x, dim=-1)/torch.arange(1, x.shape[-1]+1, device=x.device).float().view(1, 1, -1)
        
        #x = torch.cat([x, x[..., 0:1].expand_as(x)], dim=1)
        
        x_multi = self.fconv(x)
        x_single = self.fconv2(x)
        return x_single, x_multi

    def project_sum(self, x, pad_mask):
        x = F.softmax(x, dim=1)
        classes = torch.arange(-13, 4 + 1).view(1, 18, 1).float().to(x.device)
        pred = x * classes
        return (pred * pad_mask).sum(dim=1, keepdim=True)

    def forward(self, data):
        ms, pad_mask = data

        emb = self.embed(ms, pad_mask)
        emb = self.patient_embed(emb, pad_mask)

        return self.classify(emb, pad_mask)

    def make_decisions(self, mss, pad_mask, eval=True):

        if eval:
            self.eval()
        y_pred_single, y_pred_multi = self.forward((mss, pad_mask))
        self.train()

        pos = torch.tensor(Patient.utility_TP(18, 13)).view(1, 18, 1).to(y_pred_single.device)
        neg = torch.tensor(Patient.utility_FN(18, 13)).view(1, 18, 1).to(y_pred_single.device)

        ps = F.softmax(y_pred_multi, dim=1)
        sigmoids = y_pred_single.sigmoid()

        E_pos = ((pos * ps) * sigmoids).sum(dim=1, keepdim=True) + ((1 - sigmoids) * (-0.05)).sum(dim=1, keepdim=True)
        E_neg = ((neg * ps) * sigmoids).sum(dim=1, keepdim=True)

        decisions = (E_pos >= E_neg)

        return decisions.squeeze(dim=1), sigmoids.squeeze(dim=1)

    def predict_patient(self, patient: Patient):
        device = next(self.parameters()).device  # Assumption that all the model parameters are on the same device.

        batch = collate4([patient.represent()])
        mss, pad_mask, pu, nu, classes, one_hot_classes = [
            [m.to(device) for m in x] if isinstance(x, list) else x.to(device) for x in batch]

        decisions, probs = self.make_decisions(mss, pad_mask)

        return decisions.squeeze(dim=0).to("cpu").numpy()[-1], probs.squeeze(dim=0).to("cpu").numpy()[-1]