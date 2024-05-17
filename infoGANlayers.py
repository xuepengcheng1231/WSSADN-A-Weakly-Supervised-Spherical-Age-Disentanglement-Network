import torch
import torch.nn as nn
import numpy as np
import pandas
import torch.nn.functional as F
class Generator(nn.Module):
    # infogan generator
    def __init__(self,input_dim,output_dim,len_c_code):
        super(Generator,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.len_c_code = len_c_code
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.len_c_code,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 324),
            nn.BatchNorm1d(324),
            nn.LeakyReLU(0.2),
        )
    def forward(self,input):
        x = self.fc(input)
        return x

class Discriminator(nn.Module):
    def __init__(self,input_dim, output_dim,len_c_code):
        super(Discriminator,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.len_c_code = len_c_code
        self.block = nn.Conv1d(256,1,kernel_size=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )
        self.Qnet = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )

        self.discriminator = nn.Linear(32,2)
    def forward(self,input):
        if input.ndim == 3:
            input = self.block(input).view(input.shape[0],input.shape[2])
        x = self.fc(input)
        x = self.discriminator(x)
        a = x[:,0]
        b = x[:,1]
        return a,b,input


def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class SelfAttention(nn.Module):
    """
    Self attention Layer.
    Source paper: https://arxiv.org/abs/1805.08318
    Input:
        x : input feature maps( B X C X W X H)   batch*channel*width *height
    Returns :
        self attention feature maps

    """

    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.f = nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=1)
        self.g = nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=1)
        self.h = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)
    def forward(self, x):

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        attention = torch.bmm(f.permute(0, 2, 1), g)  
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)
        out = self.gamma * self_attetion + x
        return out
