import numpy

def standardize(X):
    return (X-X.mean(axis=0))/X.std(axis=0)

def normalize(X):
    return (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

class Standardizer:

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        return (X-self.mean)/self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Normalizer:

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)

    def transform(self, X):
        return (X-self.min)/(self.max-self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
