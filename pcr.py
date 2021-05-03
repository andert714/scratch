import numpy as np
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=100, n_features=20, n_informative=5, random_state=1)
k = 5 # Number of principal components used

########## Principal Component Regression
# Center data
Xs = X - X.mean(axis=0)  
ys = y - y.mean()

evalues, evectors = np.linalg.eig(Xs.transpose().dot(Xs))
eig_order = evalues.argsort()[::-1]
Lambda = np.diag(evalues[eig_order])
V = evectors[:,eig_order]
Vk = V[:,:k]

W = Xs.dot(Vk) # X*Vk
gamma_hat = np.linalg.inv(W.transpose().dot(W)).dot(W.transpose()).dot(ys)
beta_hat = Vk.dot(gamma_hat) + X.mean(axis=0)

pca = PCA(n_components=k, svd_solver='full')
pca.fit(X)

pca.components_

X_pca = pca.transform(X)

m = LinearRegression()
m.fit(X_pca, y)

