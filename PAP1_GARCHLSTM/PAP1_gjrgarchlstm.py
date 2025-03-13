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
from PAP1_utils import FinData, train_model



BATCH_SIZE = 128
HIDDEN_SIZE = 8
SEQ_LENGTH = 7
HORIZON = 1





class GARCHLSTMCell(nn.Module):
    def __init__(self, hidden_size):
        super(GARCHLSTMCell, self).__init__()
        self._raw_omega = nn.Parameter(torch.tensor(0.0))
        self._raw_alpha = nn.Parameter(torch.tensor(0.1))
        self._raw_beta  = nn.Parameter(torch.tensor(0.8))
        self._raw_gamma  = nn.Parameter(torch.tensor(0.5))
        self.Wf = nn.Linear(1 + 1, hidden_size, bias=True)
        self.Wi = nn.Linear(1 + 1, hidden_size, bias=True)
        self.Wc = nn.Linear(1 + 1, hidden_size, bias=True)
        self.w = nn.Linear(hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.sig2_prev = None # not yet set - call init before garch
        self.resids_prev = None # not yet set - call init before garch
    
    def init_garch(self, batch_size):
        self.sig2_prev = torch.zeros(batch_size, 1) # torch.Size([64, 1])
        self.resids_prev = torch.zeros(batch_size, 1) # torch.Size([64, 1])

    def forward(self, demeaned_input_tensor, hidden_sig2_tensor, hidden_memory_tensor):
        conc_tensor = torch.cat((demeaned_input_tensor, hidden_sig2_tensor), dim=1) #(batch)
        F = self.sigmoid(self.Wf(conc_tensor))
        i = self.sigmoid(self.Wi(conc_tensor))
        c_squiggle = self.tanh(self.Wc(conc_tensor))
        omega, alpha, beta, gamma = self.reparameterize_garch()
        o = self.gjr_garch_model(demeaned_input_tensor, omega, alpha, beta, gamma) #(batch, hidden)
        hidden_memory_tensor = F * hidden_memory_tensor + i * c_squiggle
        hidden_sig2_tensor = o * (1 + self.w(self.tanh(hidden_memory_tensor)))
        return hidden_sig2_tensor, hidden_memory_tensor
    
    def reparameterize_garch(self):
        raw_omega, raw_alpha, raw_beta, raw_gamma = self._raw_omega, self._raw_alpha, self._raw_beta, self._raw_gamma
        omega = torch.nn.functional.softplus(raw_omega) / 100  # => omega > 0
        alpha_unnorm = torch.exp(raw_alpha)
        beta_unnorm  = torch.exp(raw_beta)
        gamma_unnorm  = torch.exp(raw_gamma)
        sum_unnorm = 1.0 + alpha_unnorm + beta_unnorm + gamma_unnorm / 2 # Sum for alpha + beta + gamma / 2 < 1
        alpha = alpha_unnorm / sum_unnorm
        beta  = beta_unnorm  / sum_unnorm
        gamma = gamma_unnorm / sum_unnorm
        return omega, alpha, beta, gamma
    
    def view_params(self):
        omega, alpha, beta, gamma = self.reparameterize_garch()
        return omega.to(torch.half).item(), alpha.to(torch.half).item(), beta.to(torch.half).item(), gamma.to(torch.half).item()
    
    def gjr_garch_model(self, input_tensor, omega, alpha, beta, gamma):
        I = (input_tensor < 0).float()
        self.resids_t = input_tensor - mean_model(input_tensor)
        sig2_t = omega + self.resids_t**2 * (alpha + gamma * I) + self.sig2_prev * beta# (64, 1) volatilies
        self.sig2_prev = sig2_t
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

        return hidden_sig2_tensor



def mean_model(input_tensor):
    # return mu.expand_as(input_tensor)  # Ensures mu is the same shape as input_tensor while keeping gradients
    # return torch.full_like(input_tensor, mu, dtype=input_tensor.dtype, device=input_tensor.device)
    return torch.zeros_like(input_tensor)

def gaussian_nll(x, sigma2, eps=1e-6):
    sigma2_clamped = sigma2.clamp_min(eps)  # Avoid zero variance issues
    nll = 0.5 * torch.log(2 * math.pi * sigma2_clamped) + 0.5 * ((x - mean_model(x))**2 / sigma2_clamped)
    return nll # (batch_size, ...)

def student_t_nll(x, sigma2, v=5, eps=1e-6):
    sigma2_clamped = sigma2.clamp_min(eps)  # Avoid zero variance issues
    log_term = 0.5 * torch.log(sigma2_clamped)  # First log term
    ratio = ((x - mean_model(x))**2) / ((v - 2) * sigma2_clamped + eps)
    likelihood_term = ((v + 1) / 2) * torch.log(1 + ratio)
    return log_term + likelihood_term  # Final loss computation





# for t in ['^DJI', '^GSPC', '^IXIC', 'GC=F', 'EURUSD=X']:
for t in ['GC=F']:

    FD = FinData()
    prices, logreturns = FD.get_numpy_data(t, '2010-01-01', '2020-01-01')
    data = FD.get_tensor(logreturns)
    actual_vol = FD.compute_actual_vol(window=5)
    train_dataloader, test_dataloader = FD.get_loaders(SEQ_LENGTH, BATCH_SIZE, HORIZON)
    FD.get_data_table()


    # TEST MODEL
    model = GARCHLSTM(HIDDEN_SIZE)
    # X_batch, y_batch = next(iter(train_dataloader))
    # y_pred = model(X_batch)


    # TRAIN MODEL
    epochs = 2000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_model(model, epochs, student_t_nll, optimizer, train_dataloader, test_dataloader)


    FD.vanilla_garch_eval(SEQ_LENGTH)
    FD.plot_fin_data(model)