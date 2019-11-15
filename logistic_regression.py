import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris["data"]
y = (iris["target"] == 2).astype(int)

X.shape
y.shape


lr = LogisticRegression()
lr.fit(X, y)

lr.coef_
lr.intercept_


# 0
x_plt = np.linspace(0, 5)
y_plt = -np.log(x_plt)

plt.scatter(x_plt, y_plt)
plt.show()



def calc_logit_function(x):
    return 1/(1+np.exp(-x))


def calc_log_loss(y_true, prob):
    # log_loss = -(np.matmul(y_true, np.log(prob)) + np.matmul(1-y_true, np.log(1-y_true)))/y_true.shape[0]
    log_loss = -np.mean(y_true*np.log(prob) + (1-y_true)*np.log(1-prob))
    return log_loss


def calc_log_loss_derivative(theta, x, y, idx):
    return np.mean((calc_logit_function(np.matmul(theta, x.T)) - y) * x[:, idx])

X = np.concatenate((np.ones((150, 1)), X), axis=1)
X.shape
learning_rate = 0.001
n_iters = 10000
theta = np.array([0, 0, 0, 0, 0])


for i in range(n_iters):
    grads = np.array([calc_log_loss_derivative(theta, X, y, 0), calc_log_loss_derivative(theta, X, y, 1),
                      calc_log_loss_derivative(theta, X, y, 2), calc_log_loss_derivative(theta, X, y, 3),
                      calc_log_loss_derivative(theta, X, y, 4)])

    theta = theta - learning_rate*grads

theta