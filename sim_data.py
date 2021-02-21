import numpy as np
from numpy.linalg import cholesky
from scipy.stats import norm

def make_X(nside, p, intercept=True):
  n = nside**p
  x = np.linspace(0, 1, nside)
  cord_matrices = np.meshgrid(*[x]*p)
  cols = [i.reshape(n, 1) for i in cord_matrices]
  X = np.hstack(cols)
  if intercept:
    int_col = np.ones((n, 1))
    X = np.hstack((int_col, X))
  return X

def sim_y(X, beta, Sigma):
  n = X.shape[0]
  chol = cholesky(Sigma)
  Z = norm.rvs(size=(n, 1))
  y = X @ beta + chol.T @ Z
  return y