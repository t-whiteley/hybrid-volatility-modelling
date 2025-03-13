import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from arch import arch_model



import yfinance as yf
import numpy as np

def winsorize_series(series, lower_quantile=0.01, upper_quantile=0.99):
    return series
    # lower = np.quantile(series, lower_quantile)
    # upper = np.quantile(series, upper_quantile)
    # return np.clip(series, lower, upper)


class FinData():
    def __init__(self):
        self.tick = None
        self.data = None
        self.actual_vol = None

    def get_numpy_data(self, tick, d1, d2):
        self.tick = tick
        df = yf.download(tick, start=d1, end=d2)    
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        prices = df['Close'].to_numpy()
        logreturns = df['Log_Return'].to_numpy() * 100
        demeaned_logreturns = logreturns - np.mean(logreturns)
        return prices, demeaned_logreturns
    

    def get_tensor(self, timeseries):
        self.data = winsorize_series(torch.tensor(timeseries, dtype=torch.float).squeeze())
        return self.data
    
    
    def compute_actual_vol(self, window=5):
        """Computes realized volatility as per the paper: σ²_t = (mean(ϵ²_{t-k+1}, ..., ϵ²_t))"""
        arr = self.data.numpy()
        vol_series = pd.Series(arr**2).rolling(window).mean().bfill()
        self.actual_vol = winsorize_series(torch.tensor(vol_series.values, dtype=torch.float))
        return self.actual_vol


    def get_loaders(self, seq_length, batch_size, h):
        """Formats training data as per the paper: [train:(ϵ_{t-k+1}, ..., ϵ_t), test:σ²_{t+1}]*100, epst: ϵ_{t+1} (required by the loss function!!)"""
        X = torch.stack([self.data[i:i + seq_length] for i in range(len(self.data) - seq_length - (h-1))])
        y = self.actual_vol[seq_length+h-1:]
        e = self.data[seq_length+h-1:]

        self.X_train, self.y_train, self.e_train = X[:int(0.9*X.shape[0]), :], y[:int(0.9*X.shape[0])], e[:int(0.9*X.shape[0])]
        self.X_test, self.y_test, self.e_test = X[int(0.9*X.shape[0]):, :], y[int(0.9*X.shape[0]):], e[int(0.9*X.shape[0]):]
        # print(self.X_train.shape, self.y_train.shape, self.e_train.shape) #(samples, seq_length) (samples) (samples)

        self.train_dataset = TensorDataset(self.X_train, self.y_train, self.e_train)
        self.test_dataset = TensorDataset(self.X_test, self.y_test, self.e_test)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return self.train_dataloader, self.test_dataloader
    

    def vanilla_garch_eval(self, seq_length):
        X_full_np = torch.cat((self.X_train, self.X_test), dim=0)[:, -1].numpy().flatten()  # Full dataset (train + test)
        X_train_np = self.X_train[:, -1].numpy().flatten()  # Train portion only
        X_test_np = self.X_test[:, -1].numpy().flatten()    # Test portion only

        # garch_model = arch_model(X_train_np, vol='Garch', p=1, q=1)
        garch_model = arch_model(X_train_np, mean='Zero', vol='GARCH', p=1, q=1, dist='StudentsT')

        garch_fit = garch_model.fit(disp="off")  # Fit without printing
        omega, alpha, beta = garch_fit.params["omega"], garch_fit.params["alpha[1]"], garch_fit.params["beta[1]"]
        

        
        # Compute volatility recursively on the FULL dataset (train + test)
        sig2_t = np.zeros(len(X_full_np))  # Storage for predicted volatilities
        sig2_t[0] = 0

        for t in range(1, len(X_full_np)):
            sig2_t[t] = omega + alpha * (X_full_np[t-1] ** 2) + beta * sig2_t[t-1]

        variance_forecast_train = sig2_t[:len(X_train_np)]  # Crop only the test section
        variance_forecast_test = sig2_t[len(X_train_np):]  # Crop only the test section

        train_size = len(self.X_train)  # same as len(self.y_train)
        test_size  = len(self.X_test)   # same as len(self.y_test)
        train_start = seq_length
        train_end   = seq_length + train_size
        test_start  = train_end
        test_end    = train_end + test_size

        train_index = np.arange(train_start, train_end)
        test_index  = np.arange(test_start,  test_end)
        logreturns_np = self.data.numpy()
        logreturns_train = logreturns_np[train_index]
        logreturns_test  = logreturns_np[test_index]

        actual_vol_train = self.actual_vol[train_index].numpy()
        actual_vol_test  = self.actual_vol[test_index].numpy()
        # print(len(variance_forecast_train), len(train_index))
        # print(len(X_test_np), len(variance_forecast), len(actual_vol_test))

        mse_test = mean_squared_error(actual_vol_test, variance_forecast_test)
        mae_test = mean_absolute_error(actual_vol_test, variance_forecast_test)

        print("===========================================")
        print(f"Vanilla-GARCH Params:")
        print(f"Omega: {omega:.3f}, Alpha: {alpha:.3f}, Beta: {beta:.3f}")
        print(f"Test MSE: {mse_test:.3f}")
        print(f"Test MAE: {mae_test:.3f}")
        print("===========================================\n")

        # plt.style.use('dark_background')
        plt.plot(train_index, logreturns_train, label="Logreturns", color="lightgray", linestyle='-')
        plt.plot(test_index,  logreturns_test,  color="lightgray", linestyle='-')
        plt.plot(train_index, actual_vol_train, label="Actual Vol", color="blue")
        plt.plot(test_index,  actual_vol_test,  color="blue")
        plt.plot(train_index, variance_forecast_train, label="Pred Vol (Train)", color="lime")
        plt.plot(test_index,  variance_forecast_test,  label="Pred Vol (Test)", color="red")
        plt.xlabel("time index")
        plt.legend(loc="upper right")
        plt.title("Vanilla-GARCH Plot")
        plt.tight_layout()
        plt.show()



    
    def get_data_table(self):
        logreturns_np = np.array(self.data) # Convert log returns to NumPy array

        length = len(logreturns_np)
        mean_logreturns = np.mean(logreturns_np)
        std_logreturns = np.std(logreturns_np)

        adf_result = adfuller(logreturns_np) # Perform Augmented Dickey-Fuller (ADF) test
        adf_statistic, p_value = adf_result[0], adf_result[1]

        summary_table = pd.DataFrame({
            "Dataset": [self.tick],
            "Length": [length],
            "Mean": [round(mean_logreturns, 4)],
            "Sd": [round(std_logreturns, 4)],
            "ADF": [round(adf_statistic, 2)],
            "P-value": [format(p_value, ".2e")]
        })

        print(summary_table)

    

    def plot_fin_data(self, model, seq_length=7):

        with torch.no_grad():
            y_pred_train = model(self.X_train).numpy()
            y_pred_test  = model(self.X_test).numpy()

        train_size = len(self.X_train)  # same as len(self.y_train)
        test_size  = len(self.X_test)   # same as len(self.y_test)
        train_start = seq_length
        train_end   = seq_length + train_size
        test_start  = train_end
        test_end    = train_end + test_size

        train_index = np.arange(train_start, train_end)
        test_index  = np.arange(test_start,  test_end)
        logreturns_np = self.data.numpy()
        logreturns_train = logreturns_np[train_index]
        logreturns_test  = logreturns_np[test_index]

        actual_vol_train = self.actual_vol[train_index].numpy()
        actual_vol_test  = self.actual_vol[test_index].numpy()

        mse_test = mean_squared_error(actual_vol_test, y_pred_test)
        mae_test = mean_absolute_error(actual_vol_test, y_pred_test)


        print("===========================================")
        print(f"GARCH-LSTM Params:")
        print(f"THETA: {[x for x in model.garchlstm_cell.view_params()]}")
        print(f"Test MSE: {mse_test:.3f}")
        print(f"Test MAE: {mae_test:.3f}")
        print("===========================================\n")

        # plt.style.use('dark_background')
        plt.plot(train_index, logreturns_train, label="Logreturns", color="lightgray", linestyle='-')
        plt.plot(test_index,  logreturns_test,  color="lightgray", linestyle='-')
        plt.plot(train_index, actual_vol_train, label="Actual Vol", color="blue")
        plt.plot(test_index,  actual_vol_test,  color="blue")
        plt.plot(train_index, y_pred_train, label="Pred Vol (Train)", color="lime")
        plt.plot(test_index,  y_pred_test,  label="Pred Vol (Test)", color="red")
        plt.xlabel("time index")
        plt.legend(loc="upper right")
        plt.title("GARCH-LSTM Plot")
        plt.tight_layout()
        plt.show()




# def train_model(model, epochs, loss_fn, optimizer, train_dataloader, test_dataloader):
#     l_train = []
#     l_test = []
#     for epoch in range(epochs):
        
#         model.train(True)
#         train_loss = 0
#         for i, (X_batch, _, e_batch) in enumerate(train_dataloader):
#             y_pred = model(X_batch)
#             loss = loss_fn(e_batch, y_pred).mean()
#             train_loss += loss
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         train_loss /= len(train_dataloader)

#         model.train(False)
#         test_loss = 0
#         for i, (X_batch, _, e_batch) in enumerate(test_dataloader):
#             with torch.no_grad():
#                 y_pred = model(X_batch)
#                 loss = loss_fn(e_batch, y_pred).mean()
#                 test_loss += loss
#         test_loss /= len(test_dataloader)

#         l_train.append(train_loss.item())
#         l_test.append(test_loss.item())

#         if not epoch % 20:
#             print(f"Epoch {epoch}: Train Loss = {train_loss.item():.3f}, Test Loss = {test_loss.item():.3f}, Params = {[round(p, 3) for p in model.garchlstm_cell.view_params()]}")

#     plt.style.use('dark_background')
#     plt.plot(l_test, label="test")
#     plt.plot(l_train, label="train")
#     plt.legend()
#     plt.show()




def train_model(
    model, epochs, loss_fn, optimizer, 
    train_dataloader, test_dataloader,
    factor=0.5,            # LR reduction factor
    patience=5,            # LR scheduler patience (epochs with no improvement)
    early_stop_patience=20 # total epochs with no improvement before stopping
):
    """
    factor=0.5: we reduce LR by factor of 2 (i.e. multiply by 0.5).
    patience=5: if val loss hasn't improved for 5 epochs, reduce LR.
    early_stop_patience=20: if val loss hasn't improved for 20 epochs, stop training.
    """

    # Create a scheduler that reduces LR on plateau of validation loss
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',      # we're tracking a "loss" metric, so lower is better
        factor=factor, 
        patience=patience,
        threshold=1e-4,  # small threshold for measuring "no improvement"
    )

    best_val_loss = float('inf')
    not_improved_count = 0

    l_train = []
    l_test = []

    for epoch in range(epochs):
        
        ####################
        #     Training     #
        ####################
        model.train()
        train_loss_accum = 0.0
        for X_batch, _, e_batch in train_dataloader:
            # 1) Forward
            y_pred = model(X_batch)
            loss = loss_fn(e_batch, y_pred).mean()

            # 2) Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        train_loss = train_loss_accum / len(train_dataloader)

        ####################
        #   Validation     #
        ####################
        model.eval()
        test_loss_accum = 0.0
        with torch.no_grad():
            for X_batch, _, e_batch in test_dataloader:
                y_pred = model(X_batch)
                loss = loss_fn(e_batch, y_pred).mean()
                test_loss_accum += loss.item()

        test_loss = test_loss_accum / len(test_dataloader)

        l_train.append(train_loss)
        l_test.append(test_loss)

        # Update LR scheduler based on validation loss
        scheduler.step(test_loss)

        # Early Stopping check
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            not_improved_count = 0
        else:
            not_improved_count += 1
            if not_improved_count > early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Print progress every 20 epochs (or as you prefer)
        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.3f}, Test Loss = {test_loss:.3f}, Params = {[round(p, 3) for p in model.garchlstm_cell.view_params()]}")

    # Plot training and validation loss
    # plt.style.use('dark_background')
    plt.figure(figsize=(6,4))
    plt.plot(l_train, label="Train Loss")
    plt.plot(l_test,  label="Val Loss")
    plt.title("Training/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()