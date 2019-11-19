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

# lr = LogisticRegression(penalty='none', solver='newton-cg')
# lr.fit(X, y)


class OwnLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1e5, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def _calculate_logit_function(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        # initialize theta parameters
        self.theta = np.zeros((X.shape[1],))

        for i in range(int(self.n_iters)):
            # calculate error using current theta parameters
            error = self._calculate_logit_function(np.matmul(self.theta, X.T)) - y
            self.theta -= self.learning_rate*np.mean(error*X.T, axis=1)

            pred = self._calculate_logit_function(np.matmul(self.theta, X.T))
            log_loss = -np.mean(y*np.log(pred) + (1-y)*np.log(1-pred))

            if self.verbose and i % 100 == 0:
                print(f"Iteration: {i}; Log loss: {log_loss}")

        return self.theta


lr = OwnLogisticRegression(learning_rate=0.1, n_iters=1e5, fit_intercept=True, verbose=False)
lr.fit(X, y)