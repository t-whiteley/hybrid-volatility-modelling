import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


import yfinance as yf
import numpy as np


class FinData():
    def __init__(self):
        self.data = None
        self.scaler = None
        self.apply_scaler = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def get_fin_data(self, tick, d1, d2, simulate_only=False):
        if simulate_only:
            logreturns = np.random.normal(loc=0.0, scale=0.01, size=500)
            prices = [100]
            for i in range(1, 500):
                prices.append(prices[-1] * np.exp(logreturns[i-1]))
            prices = np.array(prices)
        else:
            df = yf.download(tick, start=d1, end=d2)    
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            df = df.dropna()
            prices = df['Close'].to_numpy()
            logreturns = df['Log_Return'].to_numpy()

        return prices, logreturns


    def get_tensor(self, timeseries, apply_scaler):
        self.apply_scaler = apply_scaler
        if apply_scaler:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            data_scaled = scaler.fit_transform(timeseries.reshape(-1, 1)).flatten()
            data = torch.tensor(data_scaled, dtype=torch.float).squeeze()
            self.scaler = scaler
        else:
            data = torch.tensor(timeseries, dtype=torch.float).squeeze()
        
        self.data = data
        return data


    def get_loaders(self, data, seq_length, batch_size):
        X = torch.stack([data[i:i + seq_length] for i in range(len(data) - seq_length)])
        y = data[seq_length:]
        X_train, y_train = X[:int(0.8*X.shape[0]), :], y[:int(0.8*X.shape[0])]
        X_test, y_test = X[int(0.8*X.shape[0]):, :], y[int(0.8*X.shape[0]):]
        # print(X_train.shape, y_train.shape) #([samples, seq_length])

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        X_batch, y_batch = next(iter(train_dataloader))
        # print(X_batch.shape, y_batch.shape) #([batch_size, seq_length]) torch.Size([batch_size])
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return train_dataloader, test_dataloader
    

    def compute_actual_vol(self, window=10):
        """
        Computes realized volatility as per the paper:
            σ_t = sqrt(mean(ϵ²_{t-k+1}, ..., ϵ²_t))
        
        Args:
            window (int): rolling window size for realized volatility
        
        Returns:
            self.actual_vol (torch.Tensor): realized volatility time series
        """
        # Convert log returns to NumPy
        arr = self.data.numpy()  # self.data is a 1D torch.Tensor of logreturns
        
        # Compute realized volatility
        vol_series = pd.Series(arr**2).rolling(window).mean()
        
        # Fill initial NaNs (from rolling window)
        vol_series = vol_series.bfill()
        
        # Convert back to Torch Tensor
        self.actual_vol = torch.tensor(vol_series.values, dtype=torch.float)
        
        return self.actual_vol


    def get_data_wvol(self, seq_length, batch_size):
        data = torch.stack((self.data, self.actual_vol), dim=1) # (500,2)

        X = torch.stack([data[i:i + seq_length, :] for i in range(len(data) - seq_length)])
        y = data[seq_length:, 0] # just the returns column
        X_train, y_train = X[:int(0.8*X.shape[0]), :, :], y[:int(0.8*X.shape[0])]
        X_test, y_test = X[int(0.8*X.shape[0]):, :, :], y[int(0.8*X.shape[0]):]



        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        X_batch, y_batch = next(iter(train_dataloader))

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return data, train_dataloader, test_dataloader



    def plot_fin_data(self, model, seq_length=7):
        """
        Plots two subplots:
        (1) The raw logreturns (train vs test)
        (2) The actual volatility vs. the model's forecasted volatility (train vs test)
        We assume self.actual_vol has been computed, so we can plot it as "actual volatility."

        Args:
            model: a trained model that, given X, returns a forecasted volatility (or variance).
            seq_length (int): the sequence length used in get_loaders, so we can align indices.
        """

        if not hasattr(self, 'actual_vol'):
            raise ValueError("Must call self.compute_actual_vol(...) before plotting actual vs. forecasted volatility.")

        # 1) Get the model's forecast for train + test sets
        with torch.no_grad():
            # shape: (#train_samples,)   (#test_samples,)
            y_pred_train = model(self.X_train).numpy()
            y_pred_test  = model(self.X_test).numpy()

        # 2) Reconstruct the train vs test indices for the data
        #    y_train and y_test each have length = (len(data) - seq_length)*0.8, etc.
        train_size = len(self.X_train)  # same as len(self.y_train)
        test_size  = len(self.X_test)   # same as len(self.y_test)

        # The "start" of your y train is at index `seq_length` in the full `data`,
        # so the train portion goes from [seq_length, seq_length + train_size)
        # and test from [seq_length + train_size, seq_length + train_size + test_size).
        train_start = seq_length
        train_end   = seq_length + train_size
        test_start  = train_end
        test_end    = train_end + test_size

        # 3) Prepare x-axis index arrays
        train_index = np.arange(train_start, train_end)
        test_index  = np.arange(test_start,  test_end)

        # 4) Slice the actual logreturns for train/test plotting
        #    Remember self.data is length N in total, so we can do:
        logreturns_np = self.data.numpy()
        logreturns_train = logreturns_np[train_index]
        logreturns_test  = logreturns_np[test_index]

        # 5) Slice the actual volatility (the rolling stdev) we computed
        #    We'll align them with the same indices used for train/test above
        actual_vol_train = self.actual_vol[train_index].numpy()
        actual_vol_test  = self.actual_vol[test_index].numpy()

        # 6) Build the figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        # ========== SUBPLOT 1: Logreturns for train vs test ==========
        ax1.plot(train_index, logreturns_train, label="Train Logreturns", color="tab:blue")
        ax1.plot(test_index,  logreturns_test,  label="Test Logreturns",  color="tab:orange")
        ax1.set_title("Actual Logreturns (Train vs Test)")
        ax1.legend(loc="upper right")

        # ========== SUBPLOT 2: Actual Volatility vs. Forecasted Volatility ==========
        ax2.plot(train_index, y_pred_train, label="Pred Vol (Train)", linestyle="--", color="tab:red")
        ax2.plot(test_index,  y_pred_test,  label="Pred Vol (Test)",  linestyle="--", color="tab:pink")

        ax2.plot(train_index, actual_vol_train, label="Actual Vol (Train)", color="tab:green")
        ax2.plot(test_index,  actual_vol_test,  label="Actual Vol (Test)",  color="tab:gray")


        ax2.set_title("Actual vs. Forecasted Volatility")
        ax2.set_ylabel("Volatility")
        ax2.set_xlabel("Time Index")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.show()




def train_model(model, epochs, loss_fn, optimizer, train_dataloader, test_dataloader):
    for epoch in range(epochs):
        
        model.train(True)
        train_loss = 0
        for i, (X_batch, _) in enumerate(train_dataloader):
            y_pred = model(X_batch)
            x_last = X_batch[:, -1]
            loss = loss_fn(x_last, y_pred).mean()
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)

        model.train(False)
        test_loss = 0
        for i, (X_batch, _) in enumerate(test_dataloader):
            with torch.no_grad():
                y_pred = model(X_batch)
                x_last = X_batch[:, -1]
                loss = loss_fn(x_last, y_pred).mean()
                test_loss += loss
        test_loss /= len(test_dataloader)

        if not epoch % 10:
            print(epoch, train_loss, test_loss)