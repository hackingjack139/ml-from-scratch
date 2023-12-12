import numpy as np
import matplotlib.pyplot as plt

class k_means:
    def __init__(self, k=5, n_iters=1000):
        self.k = k
        self.n_iters = n_iters
        self._clusters = [[] for _ in range(self.k)]
        self._centroids = []

    def fit(self, X):
        return self.predict(X)

    def predict(self, X):
        self.X = X
        self._n_samples, n_features = X.shape
        centroid_idxs = np.random.choice(range(self._n_samples), self.k, replace=False)
        self._centroids = [X[idx] for idx in centroid_idxs]

        for _ in range(self.n_iters):
            old_centroids = self._centroids

            self._update_cluster_labels()
            self._update_centroids()

            if self._is_merged(old_centroids):
                break
        
        return self._get_cluster_labels()

    def _update_cluster_labels(self):
        self._clusters = [[] for _ in range(self.k)]

        for idx, x in enumerate(self.X):
            distances = [self._euclidian_distance(x, centroid) for centroid in self._centroids]
            closest_idx = np.argmin(distances)
            self._clusters[closest_idx].append(idx)

    def _update_centroids(self):
        self._centroids = []
        for idx, cluster in enumerate(self._clusters):
            self._centroids.append(self._calc_centroid(cluster))

    def _calc_centroid(self, cluster):
        return np.mean([self.X[idx] for idx in cluster], axis=0)

    def _euclidian_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    def _get_cluster_labels(self):
        labels = np.empty(self._n_samples)
        for cluster_idx, cluster in enumerate(self._clusters):
            for label_idx in cluster:
                labels[label_idx] = cluster_idx
    
        return labels
    
    def _is_merged(self, old_centroids):
        distances = [self._euclidian_distance(old_centroids[idx], self._centroids[idx]) for idx in range(self.k)]
        return sum(distances) == 0
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self._clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self._centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

if __name__ == "__main__":
    np.random.seed(1)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=1
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = k_means(k=clusters)
    y_pred = k.predict(X)

    k.plot()