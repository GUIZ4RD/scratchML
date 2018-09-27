import numpy as np

class Perceptron:

    def __init__(self, learning_rate=1., n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def activation(self,z):
        return (z>=0).astype(int)

    def net_input(self, X):
        return np.dot(self.weights, X.T)+self.bias

    def predict(self, X):
        z = self.net_input(X)
        y_pred = self.activation(z)
        return y_pred

    def learning_rule(self, x,y):
        y_pred = self.predict(x)
        weights_update = (y-y_pred)*x
        bias_update = y-y_pred
        return (bias_update, weights_update)

    def fit(self, X, y):

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iter):
            for x_i, y_i in zip(X, y):
                bias_update, weights_update = self.learning_rule(x_i, y_i)
                self.weights += self.learning_rate*weights_update
                self.bias += self.learning_rate*bias_update
