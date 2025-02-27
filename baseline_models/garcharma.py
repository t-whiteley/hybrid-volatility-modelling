import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import yfinance as yf



class AR1:
    def __init__(self, y, p, q):
        self.p = p
        self.q = q
        self.y = y
        self.start = max(p, q)
        self.sig2 = 1
        self.phi = [0.8 for _ in range(p)]
        self.theta = [0.1 for _ in range(q)]
        self.e = np.zeros_like(y)
    
    def predict(self):
        self.update_e()
        res = np.zeros_like(self.y)

        for i in range(self.start, len(self.y)):
            ar_part = sum(self.phi[k] * self.y[i - k - 1] for k in range(self.p))
            ma_part = sum(self.theta[k] * self.e[i - k - 1] for k in range(self.q))
            res[i] = ar_part + ma_part + np.random.normal(0, np.sqrt(self.sig2))
        return res
    
    def update_e(self):
        e = self.e
        for i in range(self.start, len(self.y)):
            ar_part = sum(self.phi[k] * self.y[i - k - 1] for k in range(self.p))
            ma_part = sum(self.theta[k] * e[i - k - 1] for k in range(self.q))
            e[i] = self.y[i] - ar_part - ma_part
        return e
    
    def J(self, x):
        self.sig2 = x[0]
        self.phi = x[1:1+self.p]
        self.theta = x[1+self.p:]
        self.update_e()
        n = len(self.y)-self.start
        e = self.e
        ssr = np.sum(e[self.start:]**2)
        nll = 0.5 * n * np.log(2.0 * np.pi) + 0.5 * n * np.log(self.sig2) + 0.5 * ssr / self.sig2
        return nll
    
    def optim(self):
        res = minimize(
            fun=self.J,
            x0=[self.sig2] + self.phi + self.theta,
            method="L-BFGS-B",
            bounds=[(1e-4, None)] + [(None, None)] * (self.p + self.q)  # Only sig2 is constrained to be positive
        )
        print(res)
        self.sig2 = res.x[0]
        self.phi = res.x[1:1+self.p]
        self.theta = res.x[1+self.p:]
        self.update_e()

class GARCH:
    def __init__(self, y):
        self.omega = 0.01
        self.alpha = 0.2
        self.beta = 0.7
        self.y = y
        self.sig2 = np.zeros_like(y)
        self.update_sig2()
    
    def predict(self):
        self.update_sig2()
        y = [self.y[0]]
        sig2 = [self.sig2[0]]
        for i in range(1, len(self.y)):
            sig2_pred = self.omega + self.alpha * y[i-1]**2 + self.beta * sig2[i-1]
            r_pred = np.random.normal(0, np.sqrt(sig2_pred))
            y.append(r_pred)
            sig2.append(sig2_pred)
        return y, sig2
    
    
    def update_sig2(self):
        sig2 = self.sig2
        for i in range(1, len(self.y)):
            sig2[i] = self.omega + self.alpha * self.y[i-1]**2 + self.beta * sig2[i-1]    
        self.sig2 = sig2

    
    def J(self, x):
        self.omega = x[0]
        self.alpha = x[1]
        self.beta = x[2]
        self.update_sig2()  

        y2 = self.y[1:]**2
        sig2_part = self.sig2[1:]

        nll = 0.5 * np.sum(
            np.log(2.0 * np.pi) 
            + np.log(sig2_part) 
            + (y2 / sig2_part)
        )
        return nll


    
    def optim(self):
        lin = LinearConstraint(np.array([[0, 1, 1]]), 1e-4, 1-1e-4)

        res = minimize(
            fun=self.J,
            x0=[self.omega, self.alpha, self.beta],
            method="L-BFGS-B",
            bounds=[(1e-4, None), (1e-4, 1-1e-4), (1e-4, 1-1e-4)],  # Only sig2 is constrained to be positive
            constraints=[lin]
        )

        print(res)
        self.omega = res.x[0]
        self.alpha = res.x[1]
        self.beta = res.x[2]
        self.update_sig2()
    
class GARCH11ARMApq:
    def __init__(self, y, p, q):
        self.p = p
        self.q = q
        self.start = max(p, q, 1)
        self.phi = [0.8 for _ in range(p)]
        self.theta = [0.1 for _ in range(q)]
        self.omega = 0.01
        self.alpha = 0.2
        self.beta = 0.7
        self.y = y
        self.sig2 = np.zeros_like(y)
        self.e = np.zeros_like(y)
        self.update_params()

    def update_params(self):
        sig2 = self.sig2
        e = self.e
        for i in range(self.start, len(self.y)):
            ar_part = sum(self.phi[k] * self.y[i - k - 1] for k in range(self.p))
            ma_part = sum(self.theta[k] * e[i - k - 1] for k in range(self.q))
            sig2[i] = self.omega + self.alpha * e[i-1]**2 + self.beta * sig2[i-1]
            e[i] = self.y[i] - ar_part - ma_part
        self.sig2 = sig2
        self.e = e
    
    def predict(self):
        self.update_params()
        y = self.y[:self.start].tolist()
        e = self.e[:self.start].tolist()
        sig2 = self.sig2[:self.start].tolist()

        for i in range(self.start, len(self.y)):
            sig2_pred = self.omega + self.alpha * e[i-1]**2 + self.beta * sig2[i-1]
            ar_part = sum(self.phi[k] * y[i - k - 1] for k in range(self.p))
            ma_part = sum(self.theta[k] * e[i - k - 1] for k in range(self.q))
            r_pred = ar_part + ma_part + np.random.normal(0, np.sqrt(sig2_pred))
            e_pred = r_pred - ar_part - ma_part
            y.append(r_pred)
            sig2.append(sig2_pred)
            e.append(e_pred)
        return y, sig2
    
    def J(self, x):
        self.omega = x[0]
        self.alpha = x[1]
        self.beta = x[2]
        self.phi = x[3:3+self.p]
        self.theta = x[3+self.p:]
        self.update_params()  

        nll = 0.5 * np.sum(
            np.log(2.0 * np.pi) +
            np.log(self.sig2[self.start:]) +
            (self.e[self.start:]**2 / self.sig2[self.start:])
        )

        return nll
    

    
    def optim(self):
        lin = LinearConstraint(np.array([[0, 1, 1] + [0]*(self.p + self.q)]), 1e-4, 1-1e-4)

        res = minimize(
            fun=self.J,
            x0=[self.omega, self.alpha, self.beta] + self.phi + self.theta,
            bounds=[(1e-4, None), (1e-4, 1-1e-4), (1e-4, 1-1e-4)] + [(None, None)] * (self.p + self.q),
            method="SLSQP",
            constraints=[lin]
        )

        print(res)
        self.omega = res.x[0]
        self.alpha = res.x[1]
        self.beta = res.x[2]
        self.phi = res.x[3:3+self.p]
        self.theta = res.x[3+self.p:]
        self.update_params()  



# x = np.linspace(0, 5*np.pi, 100)
# r = np.sin(x)

data = yf.download('NVDA', start='2022-01-01', end='2025-01-01')
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)
logreturns = data['Log_Return'].to_numpy()

m = GARCH11ARMApq(logreturns, 3, 1)
m.optim()
sig2 = m.sig2
pred_r, pred_sig2 = m.predict()


start = m.start
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
axes[0].plot(logreturns[start:], label='logreturns')
axes[0].plot(pred_r[start:], label='logreturns predicted')
axes[0].legend()
axes[1].plot(sig2[start:], label='logreturns variance')
axes[1].plot(pred_sig2[start:], label='logreturns variance predicted')
axes[1].legend()
plt.show()