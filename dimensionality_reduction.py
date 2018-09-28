import numpy as np

def pca(X,n_components):
    cov_mat = np.cov(X.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    top_k_indices = np.argsort(eig_vals)[:-(n_components+1):-1]
    W = eig_vecs[:,top_k_indices]
    pc = np.dot(X,W)
    return pc

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        cov_mat = np.cov(X.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        top_k_indices = np.argsort(eig_vals)[:-(self.n_components+1):-1]
        self.W = eig_vecs[:,top_k_indices]

    def transform(self, X):
        return np.dot(X, self.W)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
