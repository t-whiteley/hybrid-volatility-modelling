import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


ticker = '^GSPC'
start = '2020-01-01'
end = '2025-01-01'
df = yf.download(ticker, start=start, end=end)
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()
prices = df['Close'].to_numpy()
logreturns = df['Log_Return'].to_numpy()
demeaned_logreturns = logreturns - np.mean(logreturns)


# data = torch.tensor(demeaned_logreturns, dtype=torch.float).squeeze()
# print(data.shape) #torch.Size([502])
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
data = torch.tensor(data_scaled, dtype=torch.float).squeeze()

seq_length = 5
X = torch.stack([data[i:i + seq_length] for i in range(len(data) - seq_length)])
y = data[seq_length:]
X_train, y_train = X[:int(0.8*X.shape[0]), :], y[:int(0.8*X.shape[0])]
X_test, y_test = X[int(0.8*X.shape[0]):, :], y[int(0.8*X.shape[0]):]
# print(X_train.shape, y_train.shape) #torch.Size([396, 7]) torch.Size([396])
# print(X_test.shape, y_test.shape) #torch.Size([99, 7]) torch.Size([99])

batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
X_batch, y_batch = next(iter(train_dataloader))
# print(X_batch.shape, y_batch.shape) #torch.Size([32, 7]) torch.Size([32])




class GARCHLSTMCell(nn.Module):
    def __init__(self):
        super(GARCHLSTMCell, self).__init__()
        self.THETA = nn.Parameter(torch.tensor([0, 0.1, 0.8]))
        self.Wf = nn.Linear(1 + 1, 1, bias=True) # hidden + input -> hidden
        self.Wi = nn.Linear(1 + 1, 1, bias=True)
        self.Wc = nn.Linear(1 + 1, 1, bias=True)
        self.w = nn.Linear(1, 1, bias=False) # hidden -> output
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, demeaned_input_tensor, hidden_sig2_tensor, hidden_memory_tensor):
        self.omega = nn.functional.softplus(self.THETA[0])  # Ensures w > 0
        self.alpha = torch.sigmoid(self.THETA[1])  # Ensures 0 < a < 1
        self.beta = (1 - self.alpha) * torch.sigmoid(self.THETA[2])  # Ensures 0 < b < 1 and 0 < a + b < 1
        conc_tensor = torch.cat((demeaned_input_tensor.unsqueeze(dim=1), hidden_memory_tensor), dim=1) #(batch)
        F = self.sigmoid(self.Wf(conc_tensor))
        i = self.sigmoid(self.Wi(conc_tensor))
        c_squiggle = self.tanh(self.Wc(conc_tensor))
        o = self.inferGARCH11(demeaned_input_tensor.unsqueeze(dim=1), hidden_sig2_tensor)
        hidden_memory_tensor = F * hidden_memory_tensor + i * c_squiggle
        hidden_sig2_tensor = o * (1 + self.w(self.tanh(hidden_memory_tensor)))
        return hidden_sig2_tensor, hidden_memory_tensor
    
    def inferGARCH11(self, e_tm1, sig2_tm1):
        sig2_t = self.omega + self.alpha * e_tm1*e_tm1 + self.beta * sig2_tm1  # Element-wise operations
        return sig2_t
        

class GARCHLSTM(nn.Module):
    def __init__(self):
        super(GARCHLSTM, self).__init__()
        self.garchlstm_cell = GARCHLSTMCell()
    
    def forward(self, input_tensor):
        batch_size, seq_length = input_tensor.shape
        hidden_sig2_tensor = torch.zeros(batch_size, 1, device=input_tensor.device) #(batch, hidden)
        hidden_memory_tensor = torch.zeros(batch_size, 1, device=input_tensor.device) #(batch, hidden)

        res = []
        for t in range(seq_length):
            input_t = input_tensor[:, t]  # (batch)
            hidden_sig2_tensor, hidden_memory_tensor = self.garchlstm_cell(input_t, hidden_sig2_tensor, hidden_memory_tensor)
            res.append(hidden_sig2_tensor)  # hidden_sig2_tensor is (batch, 1)

        return torch.cat(res, dim=1)  # (batch, seq_len)





model = GARCHLSTM()
GaussianNLLLoss = lambda eps, sig2: torch.mean(torch.sum(torch.log(sig2)/2 + eps**2 / (2 * sig2), dim=1))
# eps, sig2 = X_batch, model(X_batch)
# loss = GaussianNLLLoss(eps, sig2)
# print(loss.shape)
# print(loss)
    



def train(epochs):
    loss_fn = GaussianNLLLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        
        model.train(True)
        train_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            y_pred = model(X_batch)
            loss = loss_fn(X_batch, y_pred)
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
                loss = loss_fn(X_batch, y_pred)
                test_loss += loss
        test_loss /= len(test_dataloader)

        if not epoch % 10:
            print(epoch, train_loss, test_loss)
            print(model.garchlstm_cell.omega)
            print(model.garchlstm_cell.alpha)
            print(model.garchlstm_cell.beta)



train(1000)
    

with torch.no_grad():
    y_pred_train = model(X_train)[:, -1].numpy()
    y_pred_test = model(X_test)[:, -1].numpy()

future_index = np.arange(len(y_pred_train), len(y_pred_train) + len(y_pred_test))
past_index = np.arange(len(y_pred_train))
plt.plot(past_index, y_train, label="actual train")
plt.plot(past_index, y_pred_train, label="pred train")
plt.plot(future_index, y_test, label="actual test")
plt.plot(future_index, y_pred_test, label="pred test")
plt.legend()
plt.show()