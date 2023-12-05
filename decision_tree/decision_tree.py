import numpy as np
from collections import Counter

class decision_tree:
    def __init__(self, max_depth=10, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        preds = [self._traverse_tree(x, self.root) for x in X]
        return preds
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples:
            value = self._most_common(y)
            return node(value = value)
        
        best_feature, best_threshold = self._best_split(X, y, n_features)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        return node(left, right, best_threshold, best_feature)
    
    def _most_common(self, y):
        counter = Counter(y)
        most_common_label = counter.most_common()[0][0]
        return most_common_label

    def _best_split(self, X, y, n_features):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._calculate_information_gain(X[:, feature], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _split(self, X_col, threshold):
        left_idxs = np.argwhere(X_col <= threshold).flatten()
        right_idxs = np.argwhere(X_col > threshold).flatten()

        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_empty():
            return node.value
        
        if x[node.feature] > node.threshold:
            return self._traverse_tree(x, node.right)
        else:
            return self._traverse_tree(x, node.left)

    def _calculate_information_gain(self, X_col, y, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_col, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        p_l, p_r = len(left_idxs)/len(y), len(right_idxs)/len(y)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        
        child_entropy = (p_l * e_l) + (p_r * e_r)

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        ps = np.bincount(y) / len(y)
        return -np.sum([p * np.log(p + 1e-10) for p in ps])


class node:
    def __init__(self, left=None, right=None, threshold=None, feature=None, value=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.value = value

    def is_empty(self):
        return self.value is not None