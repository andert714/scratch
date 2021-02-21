import numpy as np
from sim_data import *
from numpy.linalg import inv

class Ridge:
  def __init__(self, alpha):
    self.alpha = alpha

  def fit(self, X, y):
    self.n = X.shape[0]
    self.p = X.shape[1]
    self.X = X
    self.y = y
    self.coef_ = inv(X.T @ X + np.diag(np.repeat(self.alpha, self.p))) @ X.T @ y
    self.fitted = self.X @ self.coef_
    self.res = self.y - self.fitted
    self.df = self.n - self.p
    self.s2 = float(self.res.T @ self.res)/self.df

  def predict(self, X):
    yhat = X @ self.coef_
    return yhat

X = make_X(10, 2)
beta = np.array([0, 5, 0]).reshape(-1, 1)
Sigma = np.diag(np.repeat(9, X.shape[0]))
y = sim_y(X, beta, Sigma)

m = Ridge(alpha=0.5)
m.fit(X, y)
print(m.coef_)