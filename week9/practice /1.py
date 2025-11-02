from sympy import symbols, diff
import numpy as np
import matplotlib.pyplot as plt

def sgd(X, y, lr=0.01, iterations=100, tol=1e-3):
    n, m = X.shape              # n rows (samples), m cols (features)
    thetas = np.random.rand(m, 1)
    for _ in range(iterations):     # epochs
        for _ in range(n):          # n stochastic updates per epoch
            idx = np.random.randint(n)
            xi = X[idx:idx+1]       # (1,m)
            yi = y[idx:idx+1]       # (1,1)

            grad = 2 * xi.T.dot(xi.dot(thetas) - yi)   # (m,1)
            thetas_new = thetas - lr * grad

            # check AFTER computing the update; if good, RETURN the new θ
            if np.max(np.abs(thetas_new - thetas)) < tol:
                return thetas_new
            thetas = thetas_new
    return thetas


def gd(X, y, lr=0.01, iterations=100, tol=1e-3):
    n, m = X.shape
    thetas = np.random.rand(m, 1)
    for _ in range(iterations):
        grad = (2/n) * X.T.dot(X.dot(thetas) - y)   # (m,1)
        thetas_new = thetas - lr * grad
        if np.max(np.abs(thetas_new - thetas)) < tol:
            return thetas_new       # return the UPDATED θ
        thetas = thetas_new
    return thetas

theta_gd  = gd(X, y)
theta_sgd = sgd(X, y)
print(theta_gd.T, theta_sgd.T) 
