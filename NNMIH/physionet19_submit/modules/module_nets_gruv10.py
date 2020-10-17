import torch
import torch.nn as nn
import torch.nn.functional as F

class ModGRU(nn.Module):

    def __init__(self, input_dim, h_dim):
        super(ModGRU, self).__init__()

        self.cell = nn.GRUCell(input_dim, h_dim)
        self.comb = nn.Linear(input_dim*2, input_dim)
        self.com2 = nn.Linear(input_dim+h_dim, h_dim)

    def forward(self, X, hx, M=None, mask=None):
        """
        X: [seqlen, batchsize, dim]
        """
        output = []
        if M is None:
            M = torch.ones_like(X)
        for i in range(X.size(0)):
            comb = self.comb(torch.cat([X[i], M[i]], dim=-1))
            _hx = self.cell(comb, hx)
            mask_i = mask[i].unsqueeze(-1).expand(hx.size())
            hx = _hx * mask_i + hx * (1 - mask_i)
            out = self.com2(torch.cat([comb, hx],dim=-1))
            output.append(out)
        output = torch.stack(output)
        return output, hx
        
class GRUv10(nn.Module):

    def __init__(self, args, base_list):
        super(GRUv10, self).__init__()

        i_dim = args.i_dim
        self.h_dim = args.h_dim
        #self.use_src = (args.base_list is None or "src" in args.base_list)
        if base_list is not None: base_list = base_list[1]
        self.use_src = (base_list is None or "src" in base_list)

        if self.use_src:
            self.i2h = nn.Linear(6, self.h_dim)
            self.rnn1 = ModGRU(i_dim-6, self.h_dim)
        else:
            self.i2h = nn.Linear(i_dim, self.h_dim)
            self.rnn1 = ModGRU(i_dim, self.h_dim)
        self.rnn2 = ModGRU(self.h_dim, self.h_dim)
        self.lnorm1 = nn.LayerNorm(self.h_dim)
        self.lnorm2 = nn.LayerNorm(self.h_dim)
        self.lnorm3 = nn.LayerNorm(8)
        self.nn1 = nn.Linear(self.h_dim, 8)
        self.nn2 = nn.Linear(8, 2)

    def forward(self, X, M, mask):
        """
        Input: [seqlen, batch, channel]
        Output: [seqlen, batch, channel]
        """
        bs = X.size(1)
        if self.use_src:
            patient_info = X[:, :, -6:][0]
            Xin = X[:, :, :-6]
            Min = M[:, :, :-6]
        else:
            patient_info = X[0]
            Xin, Min = X, M
        h0 = self.i2h(patient_info)
        hx, _ = self.rnn1(Xin, h0, Min, mask)
        hx = self.lnorm1(hx)        
        hx, _ = self.rnn2(hx, h0, None, mask)        
        hx = self.lnorm2(hx)
        hx = F.relu(self.nn1(hx))
        hx = self.lnorm3(hx)
        hx = self.nn2(hx)
        return hx
