# -*- coding: utf-8 -*-
# @Author: Chloe
# @Date:   2019-07-22 13:06:21
# @Last Modified by:   Chloe
# @Last Modified time: 2019-07-22 13:07:03


from torch.autograd import Variable
import torch.nn.functional as F
import torch

class CNN(torch.nn.Module):

    #Our batch shape for input x is (3, 32, 32)

    def __init__(self, input_width, input_height):


        super(CNN, self).__init__()
        self.input_height = input_height
        self.input_width = input_width

        in_channel = 1
        out_channel = 1
        padding = 0
        dilation = 1
        kernel_size = 3
        stride = 2

        # https://pytorch.org/docs/stable/nn.html#id22
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel,
                                     kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        conv1_output_height = int(((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
        conv1_output_width = int((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

        # https://pytorch.org/docs/stable/nn.html#maxpool2d
        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        pool_output_height = int(((conv1_output_height + 2 *padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
        pool_output_width = int(((conv1_output_width + 2 *padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

        # Dropout
        self.dropout = torch.nn.Dropout(0.5)
        
        # hidden_size = 64 --> can change this later
        self.fc1 = torch.nn.Linear(pool_output_height * pool_output_width, 64)

        # 64 input features, 2 output features (sepsis vs no sepsis)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):

        x = x.view(-1, 1, self.input_height, self.input_width)  # ensure correct size for input
        batch_size = x.size()[0]

        # Conv1 - activation layer
        x = F.relu(self.conv1(x))

        # Max pool
        x = self.pool(x)

        # Dropout
        x = self.dropout(x)

        # Reshape data
        x = x.view(batch_size, -1)

        # Fully connected - activation layer
        x = F.relu(self.fc1(x))

        # Output layer
        x = self.fc2(x)
        return(x)