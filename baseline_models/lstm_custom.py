# ASSUMPTIONS: INPUT DIM is 1, OUTPUT DIM is 1, NO STACKING IMPLEMENTED

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


ticker = '^GSPC'
start = '2022-01-01'
end = '2025-01-01'
df = yf.download(ticker, start=start, end=end)
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()
prices = df['Close'].to_numpy()
logreturns = df['Log_Return'].to_numpy()


data = torch.tensor(logreturns, dtype=torch.float).squeeze()
# print(data.shape) #torch.Size([502])
# scaler = MinMaxScaler(feature_range=(-1, 1))
# data_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
# data = torch.tensor(data_scaled, dtype=torch.float).squeeze()

seq_length = 7
X = torch.stack([data[i:i + seq_length] for i in range(len(data) - seq_length)])
y = data[seq_length:]
X_train, y_train = X[:int(0.8*X.shape[0]), :], y[:int(0.8*X.shape[0])]
X_test, y_test = X[int(0.8*X.shape[0]):, :], y[int(0.8*X.shape[0]):]
# print(X_train.shape, y_train.shape) #torch.Size([396, 7]) torch.Size([396])
# print(X_test.shape, y_test.shape) #torch.Size([99, 7]) torch.Size([99])

batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
X_batch, y_batch = next(iter(train_dataloader))
# print(X_batch.shape, y_batch.shape) #torch.Size([32, 7]) torch.Size([32])



class LSTMCell(nn.Module):
    def __init__(self, hidden_size):
        super(LSTMCell, self).__init__()
        self.Wf = nn.Linear(1 + hidden_size, hidden_size, bias=True)
        self.Wi = nn.Linear(1 + hidden_size, hidden_size, bias=True)
        self.Wc = nn.Linear(1 + hidden_size, hidden_size, bias=True)
        self.Wo = nn.Linear(1 + hidden_size, hidden_size, bias=True)
        self.output_transform = nn.Linear(hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_tensor, hidden_tensor, memory_tensor):
        conc_tensor = torch.cat((input_tensor.unsqueeze(dim=1), hidden_tensor), dim=1)
        F = self.sigmoid(self.Wf(conc_tensor))
        i = self.sigmoid(self.Wi(conc_tensor))
        c_squiggle = self.tanh(self.Wc(conc_tensor))
        o = self.sigmoid(self.Wo(conc_tensor))
        
        memory_tensor = F * memory_tensor + i * c_squiggle
        hidden_tensor = o * self.tanh(memory_tensor)
        output_tensor = self.output_transform(hidden_tensor)
        return output_tensor, hidden_tensor, memory_tensor


class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(hidden_size)
    
    def forward(self, input_tensor):
        batch_size, seq_length = input_tensor.shape
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        memory_state = torch.zeros(batch_size, self.hidden_size, device=input_tensor.device)
        for t in range(seq_length-1):
            input_t = input_tensor[:, t]
            _, hidden_state, memory_state = self.lstm_cell(input_t, hidden_state, memory_state)
        
        output, _, _ = self.lstm_cell(input_tensor[:, -1], hidden_state, memory_state)
        return output.squeeze()


model = LSTM(hidden_size=4)
# y_pred = model(X_batch)
# print(y_pred.shape) #torch.Size([32])
    


epochs = 200
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    
    model.train(True)
    train_loss = 0
    for i, (X_batch, y_batch) in enumerate(train_dataloader):
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)

    model.train(False)
    test_loss = 0
    for i, (X_batch, y_batch) in enumerate(test_dataloader):
        with torch.no_grad():
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            test_loss += loss
    test_loss /= len(test_dataloader)

    if not epoch % 10:
        print(epoch, train_loss, test_loss)
    


with torch.no_grad():
    y_pred_train = model(X_train).numpy()
    y_pred_test = model(X_test).numpy()

future_index = np.arange(len(y_pred_train), len(y_pred_train) + len(y_pred_test))
past_index = np.arange(len(y_pred_train))
plt.plot(past_index, y_train, label="actual train")
plt.plot(past_index, y_pred_train, label="pred train")
plt.plot(future_index, y_test, label="actual test")
plt.plot(future_index, y_pred_test, label="pred test")
plt.legend()
plt.show()