
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:50:48 2019

@author: brou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F





class LSTM(nn.Module):

    def __init__(self, nb_vital_features, nb_demo_features, nb_physio_features, hidden_dim_linear, hidden_dim_lstm, step_prediction, output_dim=2):
        super(LSTM, self).__init__()
        self.vital = nb_vital_features
        self.demo = nb_demo_features
        self.physio = nb_physio_features*2
        self.input_dim = self.vital*6 + self.demo + self.physio
        
        self.hidden_dim_lstm = hidden_dim_lstm
        self.step_pred = step_prediction
        self.nb_classes = output_dim
        self.hidden_dim_linear = hidden_dim_linear
        
        self.preprocess_vital =  nn.Linear(self.input_dim,self.input_dim*4)
        self.preprocess_vital2 =  nn.Linear(self.input_dim*4,self.input_dim*8)
        
        
        self.preprocess = nn.Linear(self.input_dim*8, self.hidden_dim_linear)


        self.lstm1 = nn.LSTM(self.hidden_dim_linear, self.hidden_dim_lstm, 1, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim_linear + self.hidden_dim_lstm, self.hidden_dim_lstm, 1, batch_first=True)
        self.lstm3 = nn.LSTM(self.hidden_dim_linear + 2* self.hidden_dim_lstm, self.hidden_dim_lstm, 1, batch_first=True)
        
        self.postprocess = nn.Linear(self.hidden_dim_linear + 3* self.hidden_dim_lstm, self.hidden_dim_linear)


        self.prediction1 = nn.Linear(self.hidden_dim_linear, self.vital*2)
        self.prediction2 = nn.Linear(self.vital*2,self.vital)

        

        self.sepsis1 = nn.Linear(self.hidden_dim_linear,self.vital*2)
        self.sepsis2 = nn.Linear(self.vital*2,self.nb_classes*self.step_pred)

        
    def forward(self, x, lens):
        
        batch_size, seq_len, _ = x.shape
        

        
        x = F.relu(self.preprocess_vital(x))
        x = F.relu(self.preprocess_vital2(x))

        
        x = F.relu(self.preprocess(x))
        inputs = torch.nn.utils.rnn.pack_padded_sequence(x,lengths=lens,batch_first=True)
        inputs1, self.hidden = self.lstm1(inputs,None)
        inputs, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs,batch_first=True)
        inputs1, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs1,batch_first=True)

        
        inputs = torch.cat((inputs1,inputs),2)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs,lengths=lens,batch_first=True)
        inputs2, self.hidden = self.lstm2(inputs,None)
        inputs, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs,batch_first=True)
        inputs2, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs2,batch_first=True)
        
        inputs = torch.cat((inputs,inputs2),2)
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs,lengths=lens,batch_first=True)
        inputs3, self.hidden =  self.lstm3(inputs,None)
        inputs, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs,batch_first=True)
        inputs3, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs3,batch_first=True)

        inputs = torch.cat((inputs,inputs3),2)



        inputs = inputs.contiguous()
        inputs = F.relu(self.postprocess(inputs))


        inputs1 = F.relu(self.prediction1(inputs))
        inputs1 = self.prediction2(inputs1)
        inputs1 = F.pad(inputs1, [0,0,0,(seq_len-inputs1.shape[1]),0,0], mode ='constant', value=0)


        outputs = F.relu((self.sepsis1(inputs)))
        outputs = F.sigmoid(self.sepsis2(outputs))
        
        outputs = outputs.view(outputs.shape[0],outputs.shape[1],self.step_pred,self.nb_classes)
        
        cum_sum = torch.cumsum(outputs[:,:,:,1],2)
        neg_cum_sum = 2*outputs[:,:,0,0].repeat(outputs.shape[2],1,1).permute(1,2,0) -torch.cumsum(outputs[:,:,:,0],2)
        outputs = torch.cat((neg_cum_sum.unsqueeze(-1),cum_sum.unsqueeze(-1)),-1)

        outputs = F.pad(outputs, [0,0,0,0,0,(seq_len-outputs.shape[1]),0,0], mode ='constant', value=0)

        
        return inputs1,outputs

   


