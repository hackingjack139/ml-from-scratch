from decision_tree import decision_tree
import numpy as np
from collections import Counter

class random_forest:
    def __init__(self, n_trees=10, max_depth=10, min_samples=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        for _ in range(self.n_trees):
            tree = decision_tree(self.max_depth, self.min_samples)
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        preds = [self._predict(x) for x in X]
        return preds

    def _predict(self, x):
        predictions = [tree.predict([x])[0] for tree in self.trees]
        return self._most_common(predictions)
    
    def _most_common(self, y):
        counter = Counter(y)
        most_common_label = counter.most_common()[0][0]
        return most_common_label

    # def predict(self, X):
    #     predictions = np.array([tree.predict(X) for tree in self.trees])
    #     tree_preds = np.swapaxes(predictions, 0, 1)
    #     predictions = np.array([self._most_common(pred) for pred in tree_preds])
    #     return predictions