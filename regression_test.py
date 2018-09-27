import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from linear_regression import LinearRegression
from metrics import mean_squared_error
import numpy as np

class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", sep='\s+',
                     names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])
boston.head()

from sklearn.preprocessing import StandardScaler

X = boston.drop("MEDV", axis=1).values
Y = boston["MEDV"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

ll = LinearRegressionGD()
ll.fit(X_train_std, Y_train)
Y_pred = ll.predict(X_test_std)

print(mean_squared_error(Y_test, Y_pred))

ll = LinearRegression()
ll.fit(X_train_std, Y_train)
Y_pred = ll.predict(X_test_std)
print(mean_squared_error(Y_test, Y_pred))

from sklearn.linear_model import LinearRegression as lr

ll = lr()
ll.fit(X_train_std, Y_train)
Y_pred = ll.predict(X_test_std)
print(mean_squared_error(Y_test, Y_pred))
