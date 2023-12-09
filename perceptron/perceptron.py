import numpy as np

class perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self._weights = None
        self._bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0

        for _ in range(self.n_iters):
            z = X @ self._weights + self._bias
            yi = self._activation_function(z)
            dw = X.T @ (yi - y)
            db = np.mean(yi - y)

            self._weights = self._weights - (self.learning_rate * dw)
            self._bias = self._bias - (self.learning_rate * db)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _activation_function(self, z):
        return np.where(z > 0, 1, 0)
    
    def _predict(self, x):
        z = x @ self._weights + self._bias
        y = self._activation_function(z)
        return y