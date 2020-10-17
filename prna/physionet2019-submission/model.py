import os
import sys
import math
import collections
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter

import utils_jr

# model: TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: seqLen x inputSize
        # transform x to dimension (N, C, L) in order to be passed into CNN
        # N: batch size; C: inputSize; L: seqLen
        output = self.tcn(x.unsqueeze(0).transpose(1,2)).transpose(1, 2)
        output = self.linear(output)
        output = self.sig(output).squeeze(0)[:,0]
        return output


class TCNRegression(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNRegression, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: seqLen x inputSize
        # transform x to dimension (N, C, L) in order to be passed into CNN
        # N: batch size; C: inputSize; L: seqLen
        output = self.tcn(x.unsqueeze(0).transpose(1,2)).transpose(1, 2)
        output = self.linear(output).squeeze(0)[:,0]
        return output

# define RITS models
class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False, device=\
        torch.device('cpu')):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.device = device
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size).to(self.device))
        self.b = Parameter(torch.Tensor(output_size).to(self.device))
        
        if self.diag:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size).to(self.device)
            self.register_buffer('m', m)  # not required as param
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)
            
    def forward(self, d):
        # Compute decay values based on delta matrix
        # gamma = exp(-(ReLU(Wd + b)))
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class RitsModel_i(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=None, \
        device=torch.device('cpu')):
        super(RitsModel_i, self).__init__()
        self.input_size = input_size   # nb_feats
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.build()
        
    def build(self):
        self.rnn_cell = nn.LSTMCell(self.input_size*2, self.hidden_size)
        
        self.temp_decay_h = TemporalDecay(self.input_size, \
            output_size=self.hidden_size, diag=False, device=self.device)
        self.temp_decay_x = TemporalDecay(self.input_size, \
            output_size=self.input_size, diag=True, device=self.device)
        
        self.hist_reg = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg = FeatureRegression(self.input_size)
        
        self.weight_combine = nn.Linear(self.input_size*2, self.input_size)
        self.out = nn.Linear(self.hidden_size, 1)
        
    def forward(self, data, direct):
        # Get data items for batch
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        # Dynamic sequence length
        seq_len = values.size()[1]

        evals = data[direct]['evals'] # gt
        eval_masks = data[direct]['eval_masks'] # 1 if value was eliminated

        # labels is now sequence (seq_len x 1), needs direct
        labels = data['labels'][direct].view(-1, 1)

        # Create an initial hidden and cell state vector for each item in batch - (#batch x #hidden)
        h = Variable(torch.zeros((values.size()[0], self.hidden_size)))
        c = Variable(torch.zeros((values.size()[0], self.hidden_size)))
        
        h, c = h.to(self.device), c.to(self.device)
        
        # init
        x_loss = 0.0
        outputs = []
        imputations = []

        for t in range(seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            # Decay the hidden vector for item in batch
            h = h * gamma_h

            # History regression: just a fully connected layer: hidden_size -> nb_feats
            x_h = self.hist_reg(h)
            # values - hist_regres_values
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            # complement vector
            x_c =  m * x +  (1 - m) * x_h
            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            # Run through the lstm cell
            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))
            
            # Compute sequence prediction loss (within loop)
            y_h = torch.sigmoid(self.out(self.dropout(h)))
            outputs.append(y_h)
        
        # Stack imputations up over time (along vertical axis)
        imputations = torch.cat(imputations, dim=1)
        
        # Stack sequence outputs
        outputs = torch.cat(outputs, dim=1)
        
        return {'loss': x_loss / seq_len, 'predictions': outputs,
                'imputations': imputations, 'labels': labels, 'evals': evals, 'eval_masks': eval_masks}


def utility_tp(t, t_optimal):
    return -0.05*(t<t_optimal-6).float() + ((t-t_optimal+6)/6)*(t>=t_optimal-6).float()*(t<=t_optimal).float() +             (1-(t-t_optimal)/9)*(t>t_optimal).float()*(t<=t_optimal+9).float() +             0*(t>t_optimal+9).float()

def utility_fn(t, t_optimal):
    return 0*(t<t_optimal).float() + (-2*(t-t_optimal)/9)*(t>=t_optimal).float()*(t<=t_optimal+9).float()             -0*(t>t_optimal+9).float()
def utility_patient(prob, label, t_optimal, device=torch.device('cpu')):
    t_vec = torch.arange(prob.size(0)).float().to(device)
    if label:
        return torch.sum(prob*utility_tp(t_vec,t_optimal)+(1-prob)*utility_fn(t_vec,t_optimal))
    else:
        return torch.sum(-0.05*prob)


class RitsModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=None, \
        device=torch.device('cpu')):
        super(RitsModel, self).__init__()
        self.input_size = input_size   # nb_feats
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device
        self.build()

    def build(self):
        self.rits_f = RitsModel_i(self.input_size, self.hidden_size, \
            self.dropout, device=self.device)
        
    def forward(self, data, lmb=100.):
        ret_f = self.rits_f(data, direct='forward')
        ret = self.get_ret(ret_f, lmb)
        return ret

    def get_ret(self, ret_f, lmb):
        '''
        Populate ret with loss, predictions and imputations 
        Computes total loss combining: impute and utility losses
        returns: ret, result as dict
        '''
        
        loss_f = ret_f['loss']
        predictions = (ret_f['predictions'])
        imputations = (ret_f['imputations'])

        # Compute utility loss
        seq_length = ret_f['labels'].shape[0]
        y = ret_f['labels'].data.cpu().numpy().reshape(-1)
        times_sepsis = sorted(np.where(y==1)[0])

        if len(times_sepsis) > 0:
            label_sepsis, t_optimal = True, times_sepsis[0]
        else:
            label_sepsis, t_optimal = False, None

        utility = utility_patient(predictions.reshape(-1), label_sepsis, t_optimal, \
            device=self.device)
        loss_u = -utility/seq_length

        # Final loss for instance
        loss = loss_f + loss_u*lmb

        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def run_on_batch(self, batch_data, optimizer):
        loss, rets = 0, []
        # Accumulate loss on batch
        for data in batch_data:
            ret = self(data, lmb=100.)
            rets.append(ret)
            loss += ret['loss']

        # Normalize loss based on batch size    
        loss = loss / len(batch_data)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item(), rets

class RNN(nn.Module):
    def __init__(self, dim_input=1, dim_hid=100, device=torch.device('cpu'), \
            survivalRNN=False, num_layer=1):
        super(RNN, self).__init__()
        self.lstm = nn.LSTMCell(dim_input, dim_hid)
        self.lstm2 = nn.LSTMCell(dim_hid, dim_hid)
        self.linear = nn.Linear(dim_hid, 1)
        self.dim_input, self.dim_hid = dim_input, dim_hid
        self.device = device
        self.survivalRNN = survivalRNN
        self.num_layer=num_layer

    def forward(self, input):
        """
        input: seqLen x inputSize
        survivalRNN: if survivalRNN=True, then treat the RNN output at each
        step as the hazard function in survival model, then the probability of
        event(having sepsis in 6 hours) is written as 1-\prod_t(1-ht)
        """
        # initialize outputs, hidden state, and cell state
        outputs = []
        h_t = torch.zeros(1, self.dim_hid, device=self.device)
        c_t = torch.zeros(1, self.dim_hid, device=self.device)
        h_t2 = torch.zeros(1, self.dim_hid, device=self.device)
        c_t2 = torch.zeros(1, self.dim_hid, device=self.device)

        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            if self.num_layer == 1:
                output = 1/(1+torch.exp(-self.linear(h_t)))
            elif self.num_layer == 2:
                output = 1/(1+torch.exp(-self.linear(h_t2)))
            outputs.append(output)
        outputs = torch.stack(outputs, 1).squeeze()
        if self.survivalRNN:
            outputs = torch.clamp(1-torch.exp(torch.cumsum(torch.log(
                1+1e-6-outputs),dim=0)), min=0, max=1)
        return outputs

class MLP(nn.Module):
    def __init__(self, dim_input=1, dim_hid=100, n_class=2, num_layer=1, \
            survivalMLP=False):
        super(MLP, self).__init__()
        self.layer = nn.ModuleList()
        self.survivalMLP = survivalMLP
        self.num_layer = num_layer
        # logistic regression
        if num_layer == 0:
            self.layer.append(nn.Linear(dim_input, n_class))
        # MLP
        elif num_layer > 0:
            self.layer.append(nn.Sequential(\
                nn.Linear(dim_input, dim_hid)))
            for i in range(num_layer-1):
                self.layer.append(nn.Sequential(\
                    nn.Linear(dim_hid, dim_hid)))
            self.layer.append(nn.Linear(dim_hid, n_class))
    
    def forward(self, x):
        """
        x: seqLen x inputSize
        """
        for i, layer in enumerate(self.layer):
            x = layer(x)
            if i < self.num_layer:
                x = nn.functional.relu(x)
        outputs = nn.functional.softmax(x, dim=1)[:,1]
        if self.survivalMLP:
            outputs = torch.clamp(1-torch.exp(torch.cumsum(torch.log(\
                    1+1e-6-outputs),dim=0)), min=0, max=1)
        return outputs


class CNN(nn.Module):
    def __init__(self, dim_input=1, n_class=2, num_layer=1, n_channel=20):
        super(CNN, self).__init__()
        self.layer = nn.ModuleList()
        self.num_layer = num_layer       
        for i in range(num_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = n_channel
            self.layer.append(nn.Sequential(\
                nn.Conv1d(in_channel, n_channel, kernel_size=3, padding=1), \
                nn.BatchNorm1d(n_channel), \
                nn.ReLU()))
        self.linear = nn.Linear(n_channel*dim_input, n_class)

    def forward(self, x):
        """
        x: seqLen x inputSize
        """
        # reshape to: seqLen x 1 x inputSize
        x = x[:,None,:]
        for i, layer in enumerate(self.layer):
            x = layer(x)
        x = self.linear(x.view(x.shape[0], -1))
        outputs = nn.functional.softmax(x, dim=1)[:,1]
        return outputs
