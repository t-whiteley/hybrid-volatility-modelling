# TODO
#  - IMPROVE INPUT X -> log returns, technical factors? multi-dim?
#  - WHAT IS SEQ_LENGTH IN THE PAPER??
#  - IMPLEMENT MEAN PREDICTOR G
#  - IMPLEMENT leptokurtic errors

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import FinData, train_model




# IMPORT DATA
BATCH_SIZE = 64
HIDDEN_SIZE = 10
SEQ_LENGTH = 7

FD = FinData()
prices, logreturns = FD.get_fin_data('^GSPC', '2022-01-01', '2025-01-01', simulate_only=False)
data = FD.get_tensor(logreturns, apply_scaler=False)
train_dataloader, test_dataloader = FD.get_loaders(data, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
FD.compute_actual_vol(window=10)



class SIGMACell(nn.Module):
    def __init__(self, hidden_size):
        super(SIGMACell, self).__init__()
        self.sig2_prev = None # not yet set - call init before garch
        self.resids_prev = None # not yet set - call init before garch
        self.recurrent_gate = nn.Linear(1 + hidden_size, hidden_size, bias=True)
        self.output_transform = nn.Linear(hidden_size, 3, bias=True)
        self.adj_softplus = AdjSoftplus(1)
        self.relu = nn.ReLU()

    def forward(self, input_tensor, hidden_tensor):
        conc_tensor = torch.cat((input_tensor, hidden_tensor), dim=1) #(batch_size, in/out_features)
        hidden_tensor = self.adj_softplus(self.recurrent_gate(conc_tensor))
        output_tensor = self.relu(self.output_transform(hidden_tensor)) # torch.Size([64, 2])
        sig2_t = self.garch_model(input_tensor, output_tensor) # CHECK SHAPES
        return sig2_t, hidden_tensor
    
    def init_garch(self, batch_size):
        self.sig2_prev = torch.zeros(batch_size, 1) # torch.Size([64, 1])
        self.resids_prev = torch.zeros(batch_size, 1) # torch.Size([64, 1])
    
    def garch_model(self, input_tensor, output_tensor):
        alpha_t = output_tensor[:, :1] # torch.Size([64, 1])
        beta_t = output_tensor[:, 1:2] # torch.Size([64, 1])
        constant_t = output_tensor[:, 2:] # torch.Size([64, 1])
        # print(self.sig2_prev.shape, self.resids_prev.shape)
        sig2_t = self.sig2_prev * alpha_t + self.resids_prev**2 * beta_t + constant_t # (64, 1) volatilies
        self.sig2_prev = sig2_t
        self.resids_prev = input_tensor - mean_model(input_tensor)
        return sig2_t
    

class GARCH_RNN(nn.Module):
    def __init__(self, hidden_size):
        super(GARCH_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = SIGMACell(hidden_size)
    
    def forward(self, input_tensor):
        batch_size, seq_length = input_tensor.shape
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        self.rnn_cell.init_garch(batch_size)

        for t in range(seq_length-1):
            input_t = input_tensor[:, t:t+1]  # (batch_size, 1)
            _, hidden_state = self.rnn_cell(input_t, hidden_state)
        
        sig2_t, hidden_state = self.rnn_cell(input_tensor[:, -1:], hidden_state)
        self.hidden_save = hidden_state
        return sig2_t.squeeze()
    

class AdjSoftplus(nn.Module):
    def __init__(self, beta):
        super(AdjSoftplus, self).__init__()
        self.beta = beta

    def forward(self, x):
        beta = torch.full_like(x, self.beta)
        s = lambda scalar: torch.full_like(x, scalar)
        basic = 1/beta * torch.log(1 + torch.exp(beta * x))
        denom = 1 / beta * torch.log(1 + torch.exp(beta) - torch.log(s(2))/beta)
        return torch.max(s(0), basic / denom)


def mean_model(input_tensor):
    return torch.zeros(input_tensor.shape[0], 1) # torch.Size([64, 1])
    
def gaussian_nll_zero_mean(x, sigma2, eps=1e-6):
    sigma2_clamped = sigma2.clamp_min(eps)
    # 0.5 * log(2*pi*sigma2) + x^2/(2*sigma2)
    nll = 0.5 * torch.log(2 * math.pi * sigma2_clamped) + 0.5 * ((x - mean_model(x))**2 / sigma2_clamped)
    return nll # (batch_size, ...)



# TEST MODEL
model = GARCH_RNN(HIDDEN_SIZE) # in the paper its 10
X_batch, y_batch = next(iter(train_dataloader))
y_pred = model(X_batch)
# print(y_pred.shape) #(batch_size)

# TRAIN MODEL
epochs = 100
loss_fn = gaussian_nll_zero_mean # loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, epochs, loss_fn, optimizer, train_dataloader, test_dataloader)


FD.plot_fin_data(model)