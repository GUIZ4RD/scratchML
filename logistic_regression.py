import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=.05, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def activation(self,z):
        return 1. / (1. + np.exp(-z))

    def net_input(self, X):
        return np.dot(self.weights, X.T)+self.bias

    def predict(self, X):
        z = self.net_input(X)
        a = self.activation(z)
        y_pred = (a>=0.5).astype(int)

        return y_pred


    def derivate_costfunc(self, X, y):
        z = self.net_input(X)
        a = self.activation(z)
        errors = (y - a)
        dw = X.T.dot(errors)
        db = errors.sum()
        return (dw,db)

    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iter):
            dw, db = self.derivate_costfunc(X, y)
            self.weights += self.learning_rate*dw
            self.bias += self.learning_rate*db
