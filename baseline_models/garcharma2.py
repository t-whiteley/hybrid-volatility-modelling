import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import deque
from scipy.optimize import differential_evolution, LinearConstraint


class ARMA11:
    def __init__(self, data):
        self.mu = np.average(data)
        self.phi = 0.05
        self.theta = -0.05
        self.r = data
        self.eps = np.zeros_like(data)
        self.build_eps()
        self.sig2 = 1.0
    
    def build_eps(self):
        eps = np.zeros_like(self.r)
        for t in range(1, len(self.r)):
            eps[t] = self.r[t] - (self.mu + self.phi * self.r[t-1] + self.theta * eps[t-1])

        self.eps = eps


    def infer(self, n):
        r_prev, eps_prev = self.r[-1], self.eps[-1]
        r = []
        for _ in range(n):
            # eps_t = 0
            eps_t = np.random.normal(0, np.sqrt(self.sig2))
            r_t = self.mu + self.phi * r_prev + self.theta * eps_prev + eps_t

            r_prev = r_t
            eps_prev = eps_t
            r.append(r_t)
        return np.array(r)

    def infer_given_eps(self, n, eps):
        r_prev, eps_prev = self.r[-1], self.eps[-1]
        r = []
        for i in range(n):
            eps_t = eps[i]
            r_t = self.mu + self.phi * r_prev + self.theta * eps_prev + eps_t

            r_prev = r_t
            eps_prev = eps_t
            r.append(r_t)
        return np.array(r)
    

    def unload_params(self, x):
        self.mu, self.phi, self.theta, self.sig2 = x
        self.build_eps()

    def normal_nll(self, x):
        self.unload_params(x)
        if (self.sig2 < 0): return 1e10

        conditional = np.sqrt(self.sig2)
        likelihood = 1 / (np.sqrt(2 * np.pi) * conditional) * np.exp(-self.eps**2 / (2 * conditional**2))
        nll = -np.sum(np.log(likelihood))
        # print(x, nll)
        return nll
    
    def fit(self):

        bounds = [(-100,100), (-1,1), (-1,1), (1e-4, 1e10)]
        
        res = differential_evolution(
            self.normal_nll,
            bounds,
        )

        print(res.success, res.message)
        print(res.x)
        self.unload_params(res.x)

class GARCH11:
    def __init__(self, data):
        self.omega = 1e-6
        self.alpha = 0.001
        self.beta = 0.95
        self.r = data
        
        self.mu = np.average(data)
        self.u = self.r - self.mu
        self.sig2 = np.zeros_like(self.r)
        self.update_sig2()
    
    def update_sig2(self):
        sig2 = np.zeros_like(self.r)
        sig2[0] = np.var(self.r) # average
        for t in range(1, len(self.r)):
            sig2[t] = self.omega + self.alpha * self.u[t-1]**2 + self.beta * sig2[t-1]
        self.sig2 = sig2    
    
    def infer(self, n):
        u_prev, sig2_prev = self.u[-1], self.sig2[-1]
        r, sig2 = [], []
        for _ in range(n):
            # eps_t = 0 #
            eps_t = np.random.randn()
            sig2_t = self.omega + self.alpha*u_prev**2 + self.beta*sig2_prev # COND VAR EQUATION
            u_t = np.sqrt(sig2_t) * eps_t # ERROR EQUATION
            r_t = self.mu + u_t # MEAN EQUATION
            
            u_prev = u_t
            sig2_prev = sig2_t
            r.append(r_t)
            sig2.append(sig2_t)

        return np.array(r), np.array(sig2)

    def unload_params(self, theta):
        self.omega, self.alpha, self.beta = theta
        self.update_sig2()

    def normal_nll(self, theta):
        self.unload_params(theta)
        if (self.sig2 < 0).any():
            return 1e10

        conditional = np.sqrt(self.sig2)
        likelihood = 1 / (np.sqrt(2 * np.pi) * conditional) * np.exp(-self.u**2 / (2 * conditional**2))
        nll = -np.sum(np.log(likelihood))
        # print(theta, nll)
        return nll



    def fit(self):
        d = 1e-8
        lin1 = LinearConstraint(np.array([[0, 1, 1]]), -np.inf, 1.0-d)
        lin2 = LinearConstraint(np.array([[1, 0, 0]]), d, np.inf)
        lin3 = LinearConstraint(np.array([[0, 1, 0]]), d, np.inf)
        lin4 = LinearConstraint(np.array([[0, 0, 1]]), d, np.inf)

        bounds = [(0,1000), (0,1), (0,1)]
        res = differential_evolution(
            self.normal_nll,
            bounds,
            constraints=(lin1, lin2, lin3, lin4)
        )

        print(res.success, res.message)
        print(res.x)
        self.unload_params(res.x)


def plot1(prices, logreturns, forecast_logreturns, sig2_logreturns=[], forecast_sig2_logreturns=[]):
    initial_price = prices[-1]
    cumulative_log_returns = np.cumsum(forecast_logreturns)
    forecast_prices = initial_price * np.exp(cumulative_log_returns)

    n_obs = len(logreturns[1:])
    future_index = np.arange(n_obs, n_obs + len(forecast_logreturns))

    if len(sig2_logreturns):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    # 1) Top subplot: Historical prices + forecast prices
    axes[0].plot(np.arange(len(prices)), prices, color='green', label='Historical Prices')
    axes[0].plot(future_index, forecast_prices, linestyle='--', color='gray', label='Forecast Prices')
    axes[0].set_title('Prices (Historical + Forecast)')
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # 2) Middle subplot: Historical log returns + forecast log returns
    axes[1].plot(logreturns[1:], color='blue', label='Log Returns')
    axes[1].plot(future_index, forecast_logreturns, linestyle='--', color='gray', label='Forecast Log Returns')
    axes[1].set_title('Log Returns with Forecast')
    axes[1].set_ylabel('Return')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    if len(sig2_logreturns):
        # 3) Bottom subplot: Historical conditional variance + forecast variance
        axes[2].plot(sig2_logreturns[1:], color='red', label='Conditional Variance')
        axes[2].plot(future_index, forecast_sig2_logreturns, linestyle='--', color='gray', label='Forecast Variance')
        axes[2].set_title('Conditional Variance of Log Returns with Forecast')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Variance')
        axes[2].legend(loc='upper right')
        axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def plot2(prices, logreturns, forecast_logreturns, residuals, forecast_residuals, sig2_residuals, forecast_sig2_residuals):
    initial_price = prices[-1]
    cumulative_log_returns = np.cumsum(forecast_logreturns)
    forecast_prices = initial_price * np.exp(cumulative_log_returns)

    n_obs = len(logreturns[1:])
    future_index = np.arange(n_obs, n_obs + len(forecast_logreturns))

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

    axes[0].plot(np.arange(len(prices)), prices, color='green', label='Historical Prices')
    axes[0].plot(future_index, forecast_prices, linestyle='--', color='gray', label='Forecast Prices')
    axes[0].set_title('Prices (Historical + Forecast)')
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)

    axes[1].plot(logreturns[1:], color='blue', label='Log Returns')
    axes[1].plot(future_index, forecast_logreturns, linestyle='--', color='gray', label='Forecast Log Returns')
    axes[1].set_title('Log Returns with Forecast')
    axes[1].set_ylabel('Return')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    axes[2].plot(residuals[1:], color='red', label='ARMA Residuals')
    axes[2].plot(future_index, forecast_residuals, linestyle='--', color='gray', label='Forecast ARMA Residuals')
    axes[2].set_title('Residuals of Log Return ARMA Model with Forecast')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Residual')
    axes[2].legend(loc='upper left')
    axes[2].grid(True)

    axes[3].plot(sig2_residuals[1:], color='red', label='GARCH Conditional Variance of Residuals')
    axes[3].plot(future_index, forecast_sig2_residuals, linestyle='--', color='gray', label='Forecast Residual Variance')
    axes[3].set_title('Conditional Variance of ARMA Residuals with GARCH forecast')
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Variance')
    axes[3].legend(loc='upper left')
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()





# Download data
ticker = 'AAPL'
start = '2020-01-01'
end = '2025-01-01'
data = yf.download(ticker, start=start, end=end)
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data = data.dropna()
prices = data['Close'].to_numpy()
logreturns = data['Log_Return'].to_numpy() # input data to garch
forecast_days = 251 * 3




# # ARMA11 FITTING
# model = ARMA11(logreturns)
# model.fit()
# forecast_logreturns = model.infer(forecast_days)
# plot1(prices, logreturns, forecast_logreturns)
# # GARCH11 FITTING
# model = GARCH11(logreturns)
# model.fit()
# sig2_logreturns = model.sig2 # fitted model assumption of sig2
# infer = model.infer(forecast_days)
# forecast_logreturns = infer[0]
# forecast_sig2_logreturns = infer[1]
# plot1(prices, logreturns, forecast_logreturns, sig2_logreturns, forecast_sig2_logreturns)




# FIT LOGRETURNS TO ARMA, FIT GARCH TO ARMA RESIDUALS
model1 = ARMA11(logreturns)
model1.fit()
forecast_logreturns1 = model1.infer(forecast_days)
garchinp = model1.eps

model2 = GARCH11(garchinp)
model2.fit()
sig2_garchinp = model2.sig2
infer = model2.infer(forecast_days)
forecast_garchinp = infer[0]
forecast_sig2_garchinp = infer[1]

forecast_logreturns2 = model1.infer_given_eps(forecast_days, forecast_garchinp)
plot2(prices, logreturns, forecast_logreturns2, garchinp, forecast_garchinp, sig2_garchinp, forecast_sig2_garchinp)