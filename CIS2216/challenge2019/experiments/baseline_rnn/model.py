import torch
from torch import nn
import torch.nn.functional as F


def init_weidghts(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class MyRNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
        )
        # self.residual_mlp = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.Dropout(p=0.1)
        # )

        self.gru = nn.LSTM(input_dim, input_dim,
                           num_layers=2, batch_first=True)
        self.output = nn.Linear(input_dim, 2)
        # self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 8]), ignore_index=-1)
        self.loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1, 8]), ignore_index=-1)

        # init
        self.apply(init_weidghts)

    def forward(self, input):
        o, _ = self.gru(self.mlp(input))  # o.size=(B,L,dim)
        # residual_o = F.relu(torch.cat([self.residual_mlp(input), o],dim=-1))
        logits = self.output(F.relu(o))

        return logits

    def get_loss(self, logits, label):
        loss = self.loss(logits.view(-1, 2), label.view(-1))
        return loss
