import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=.001, n_iter=20):
        self.learning_rate = learning_rate
        self.n_iter = n_iter


    def predict(self, X):
        return np.dot(X, self.weights)+self.bias


    def derivate_costfunc(self, X, y):
        y_pred = self.predict(X)
        errors = (y - y_pred)
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
