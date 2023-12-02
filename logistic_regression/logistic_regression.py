import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class logistic_regression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            z_pred = self.X_train @ self.weights + self.bias
            y_pred = sigmoid(z_pred)
            dw = 1/n_samples * self.X_train.T @ (y_pred - y)
            db = 1/n_samples * np.sum(y_pred - y)

            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)

        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias