import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)

X.shape
y.shape


lr = LogisticRegression(penalty='none', solver='newton-cg')
lr.fit(X, y)


lr.intercept_
lr.coef_

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

learning_rate = 0.1
n_iters = 20000
theta = np.array([0, 0, 0, 0, 0])


for i in range(n_iters):
    grads = []
    for param in range(X.shape[1]):
        errors = []
        for idx in range(X.shape[0]):
            error = calc_logit_function(np.matmul(theta, X[idx, :])) - y[idx]
            errors.append(error*X[idx, param])
        grads.append(np.mean(np.array(errors)))

    theta = theta - learning_rate*np.array(grads)
    log_loss = np.mean((y*np.log(calc_logit_function(np.matmul(theta, X.T)))) + ((1-y)*np.log(1 - calc_logit_function(np.matmul(theta, X.T)))))
    print(f"Iteration: {i}; Log loss: {log_loss}")


theta

lr1 = LogisticRegression(lr=0.1, num_iter=20000, fit_intercept=True)
lr1.fit(X, y)

lr.intercept_
lr.coef_



lr.intercept_

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if (self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
        return self.theta

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold