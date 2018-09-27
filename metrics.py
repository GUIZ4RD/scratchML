import numpy as np

def accuracy(y_true, y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

def sum_squared_errors(y_true, y_pred):
    squared_errors = np.power(y_true-y_pred,2)
    return np.sum(squared_errors)

def mean_squared_error(y_true, y_pred):
    sse = sum_squared_errors(y_true, y_pred)
    return sse/len(y_true)
