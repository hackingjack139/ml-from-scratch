import numpy as np
from collections import Counter

def euclidian_distance(x, y):
    return np.sqrt(np.sum((x-y)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        indices = np.argsort(distances)[:self.k]
        nearest_neighbours = [self.y_train[i] for i in indices]

        most_common = Counter(nearest_neighbours).most_common()
        return most_common[0][0]