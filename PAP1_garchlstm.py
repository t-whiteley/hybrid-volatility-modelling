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
data = FD.get_tensor(logreturns, apply_scaler=False)
train_dataloader, test_dataloader = FD.get_loaders(data, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE)
FD.compute_actual_vol(window=10)




class GARCHLSTMCell(nn.Module):
    def __init__(self, hidden_size):
        super(GARCHLSTMCell, self).__init__()
        self.THETA = nn.Parameter(torch.tensor([0, 0.1, 0.8]))
        self.THETA.register_hook(self._apply_constraints)
        self.Wf = nn.Linear(1 + 1, hidden_size, bias=True)
        self.Wi = nn.Linear(1 + 1, hidden_size, bias=True)
        self.Wc = nn.Linear(1 + 1, hidden_size, bias=True)
        self.w = nn.Linear(hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.sig2_prev = None # not yet set - call init before garch
        self.resids_prev = None # not yet set - call init before garch

    def _apply_constraints(self, grad):
        self.THETA.data[0] = nn.functional.softplus(self.THETA.data[0]) # Ensure first value is greater than zero using softplus (smooth and differentiable)
        exp_vals = torch.exp(self.THETA.data[1:])         # Compute exp(theta_2) and exp(theta_3) for positivity
        sum_exp = 1 + exp_vals.sum()  # Normalization factor
        self.THETA.data[1:] = exp_vals / sum_exp  # Normalize to enforce sum constraint
        return grad  # Return grad unchanged for backward pass
    
    def init_garch(self, batch_size):
        self.sig2_prev = torch.zeros(batch_size, 1) # torch.Size([64, 1])
        self.resids_prev = torch.zeros(batch_size, 1) # torch.Size([64, 1])

    def forward(self, demeaned_input_tensor, hidden_sig2_tensor, hidden_memory_tensor):
        conc_tensor = torch.cat((demeaned_input_tensor, hidden_sig2_tensor), dim=1) #(batch)
        F = self.sigmoid(self.Wf(conc_tensor))
        i = self.sigmoid(self.Wi(conc_tensor))
        c_squiggle = self.tanh(self.Wc(conc_tensor))
        o = self.garch_model(demeaned_input_tensor) #(batch, hidden)
        hidden_memory_tensor = F * hidden_memory_tensor + i * c_squiggle
        hidden_sig2_tensor = o * (1 + self.w(self.tanh(hidden_memory_tensor)))
        return hidden_sig2_tensor, hidden_memory_tensor
    
    def garch_model(self, input_tensor):
        omega, alpha, beta = self.THETA
        sig2_t = self.sig2_prev * beta + self.resids_prev**2 * alpha + omega # (64, 1) volatilies
        self.sig2_prev = sig2_t
        self.resids_prev = input_tensor - mean_model(input_tensor)
        return sig2_t


class GARCHLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(GARCHLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.garchlstm_cell = GARCHLSTMCell(hidden_size)
    
    def forward(self, input_tensor):
        batch_size, seq_length = input_tensor.shape
        hidden_sig2_tensor = torch.zeros(batch_size, 1, device=input_tensor.device) #(batch, hidden)
        hidden_memory_tensor = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device) #(batch, hidden)
        self.garchlstm_cell.init_garch(batch_size)

        for t in range(seq_length):
            input_t = input_tensor[:, t:t+1]  # (batch, 1)
            hidden_sig2_tensor, hidden_memory_tensor = self.garchlstm_cell(input_t, hidden_sig2_tensor, hidden_memory_tensor)

        return hidden_sig2_tensor.squeeze()
    


def mean_model(input_tensor):
    return torch.zeros(input_tensor.shape[0], 1) # torch.Size([64, 1])
    
def gaussian_nll_zero_mean(x, sigma2, eps=1e-6):
    sigma2_clamped = sigma2.clamp_min(eps)
    nll = 0.5 * torch.log(2 * math.pi * sigma2_clamped) + 0.5 * ((x - mean_model(x))**2 / sigma2_clamped)
    return nll # (batch_size, ...)



# TEST MODEL
model = GARCHLSTM(HIDDEN_SIZE)
X_batch, y_batch = next(iter(train_dataloader))
y_pred = model(X_batch)
print(y_pred.shape) #(batch_size)

# TRAIN MODEL
epochs = 500
loss_fn = gaussian_nll_zero_mean # loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, epochs, loss_fn, optimizer, train_dataloader, test_dataloader)
print(model.garchlstm_cell.THETA)


FD.plot_fin_data(model)