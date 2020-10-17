import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nb_layer=2, add_mask=False):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nb_layer = nb_layer
        self.add_mask = add_mask

        if add_mask:
            input_size = input_size * 2

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=nb_layer,
            dropout=0.3,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )

        self.activation = nn.Softmax(dim=-1)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.nb_layer, batch_size, self.hidden_size)).to(device)

    def forward(self, batch):
        # (batch_size, sequence_length, input_size, 2)
        x, x_lengths = batch
        inputs = x[:, :, :, 0]
        masks = x[:, :, :, 1]

        # reset hidden state
        h0 = self.init_hidden(batch_size=inputs.size(0))

        # add mask indicator of missingness
        if self.add_mask:
            inputs = torch.cat((inputs, masks), dim=2)

        # Pack
        # not to end up doing more computations than required
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        input_pack = pack_padded_sequence(inputs, x_lengths, batch_first=True)

        # GRU
        outputs, _ = self.rnn(input_pack, h0)

        # Unpack
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # reshape data -> (batch_size * seq_len, nb_rnn_units)
        outputs = outputs.contiguous().view(-1, outputs.size(2))

        # decision layer
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)

        # reshape -> (batch_size, seq_len, output_dmi)
        outputs = outputs.view(inputs.size(0), x_lengths[0], self.output_size)

        # last output
        last_outputs = [outputs[i, x_lengths[i]-1, :] for i in range(len(x_lengths))]
        last_outputs = torch.stack(last_outputs)

        return last_outputs
