import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self._weights = None
        self._bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        self._bias = 0
        y = np.where(y > 0, 1, -1)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                if (y[idx] * (x @ self._weights - self._bias) >= 1):
                    self._weights -= self.learning_rate * (2 * self.lambda_param * self._weights)
                else:
                    self._weights -= self.learning_rate * ((2 * self.lambda_param * self._weights) - (y[idx] * x))
                    self._bias -= self.learning_rate * y[idx]

    def predict(self, X):
        z = X @ self._weights - self._bias
        return np.sign(z)