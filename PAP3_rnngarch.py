# TODO
#  - THERE IS NO SEQ_LENGTH!!! - MAYBE IN NONE OF THE PAPERS



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
prices, logreturns = FD.get_fin_data('^GSPC', '2022-01-01', '2025-01-01', simulate_only=True)
FD.get_tensor(logreturns, apply_scaler=False)
FD.compute_actual_vol(window=10)
data, train_dataloader, test_dataloader = FD.get_data_wvol(SEQ_LENGTH, BATCH_SIZE)
X_batch, y_batch = next(iter(train_dataloader))




class RNNCell(nn.Module):
    def __init__(self, hidden_size):
        super(RNNCell, self).__init__()
        self.recurrent_gate = nn.Linear(2 + hidden_size, hidden_size, bias=True)
        self.output_transform = nn.Linear(hidden_size, 1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, input_tensor, hidden_tensor):
        conc_tensor = torch.cat((input_tensor, hidden_tensor), dim=1) # (batch, 2+hidden_size)
        hidden_tensor = self.tanh(self.recurrent_gate(conc_tensor))
        output_tensor = self.output_transform(hidden_tensor)
        return output_tensor, hidden_tensor


class RNN(nn.Module):
    def __init__(self, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCell(hidden_size)

        self.sig2_prev = None
        self.resids_prev = None
        self.alpha = 0.2
        self.beta = 0.7

    
    def forward(self, input_tensor):
        batch_size, seq_length, features = input_tensor.shape
        self.init_garch(batch_size)
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)

        for t in range(seq_length-1):
            input_t = input_tensor[:, t, :]
            output, hidden_state = self.rnn_cell(input_t, hidden_state)
            self.garch_model(input_t[:, 0:1], omega=output)
        
        output, hidden_state = self.rnn_cell(input_tensor[:, -1], hidden_state)
        sig2 = self.garch_model(input_tensor[:, -1, 0:1], omega=output)
        return sig2.squeeze()
    
    def init_garch(self, batch_size):
        self.sig2_prev = torch.zeros(batch_size, 1) # torch.Size([64, 1])
        self.resids_prev = torch.zeros(batch_size, 1) # torch.Size([64, 1])
        
    def garch_model(self, x, omega):
        sig2_t = self.sig2_prev * self.beta + self.resids_prev**2 * self.alpha + omega # (64, 1) volatilies
        self.sig2_prev = sig2_t
        self.resids_prev = x - mean_model(x)
        return sig2_t


def mean_model(x):
    return torch.zeros(x.shape[0], 1) # torch.Size([64, 1])





# def gaussian_nll_zero_mean(x, sigma2, eps=1e-6):
#     sigma2_clamped = sigma2.clamp_min(eps)
#     # 0.5 * log(2*pi*sigma2) + x^2/(2*sigma2)
#     nll = 0.5 * torch.log(2 * math.pi * sigma2_clamped) + 0.5 * ((x - mean_model(x))**2 / sigma2_clamped)
#     return nll # (batch_size, ...)



# TEST MODEL
model = RNN(hidden_size=10)
X_batch, y_batch = next(iter(train_dataloader))
y_pred = model(X_batch)
print(y_pred.shape) #torch.Size([32])
    

# # TRAIN MODEL
# epochs = 100
# loss_fn = gaussian_nll_zero_mean # loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# train_model(model, epochs, loss_fn, optimizer, train_dataloader, test_dataloader)


FD.plot_fin_data(model)