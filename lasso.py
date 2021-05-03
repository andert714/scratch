import numpy as np
from sim_data import *
from numpy.linalg import inv
from scipy.optimize import minimize

X = make_X(10, 2)
beta = np.array([0, 5, 0]).reshape(-1, 1)
Sigma = np.diag(np.repeat(9, X.shape[0]))
y = sim_y(X, beta, Sigma)
alpha = 0.5

# Lasso loss function


# Coordinate descent
class MyLasso:
  def __init__(self, alpha, init=None):
    self.alpha = alpha
    self.init = init

  def fit(self, X, y):
    self.X = X
    self.y = y
    self.n = self.X.shape[0]
    self.p = X.shape[1]
    
    if not self.init:
      self.init = inv(self.X.T @ self.X) @ self.X.T @ self.y

    def loss(beta, X, y, alpha):
      n = self.X.shape[0]
      sse = np.sum((y - X @ beta)**2)
      l1 = np.sum(np.abs(beta[1:]))
      return sse/n + alpha*l1

    opt = minimize(loss, self.init, args=(self.X, self.y, self.alpha), method='Nelder-Mead')

    if opt.success:
      self.coef_ = opt.x
    else:
      print(opt.message)
      break

    

  
X = make_X(10, 2)
beta = np.array([0, 5, 0]).reshape(-1, 1)
Sigma = np.diag(np.repeat(9, X.shape[0]))
y = sim_y(X, beta, Sigma)
alpha = 0.5

mym = MyLasso(alpha=alpha)
mym.fit(X, y)
print(mym.coef_)


