import numpy as np

class naive_bayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._prior = np.zeros(n_classes)
        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self._classes):
            X_rows = X[y==cls]
            self._prior[idx] = X_rows.shape[0]/n_samples
            self._mean[idx, :] = np.mean(X_rows, axis=0)
            self._var[idx, :] = np.var(X_rows, axis=0)

    def predict(self, X):
        return [self._predict(x) for x in X]
    
    def _predict(self, x):
        preds = []
        
        for idx, cls in enumerate(self._classes):
            prior = self._prior[idx]
            posterior = np.sum(np.log(self._pdf(x, self._mean[idx], self._var[idx])))
            posterior = posterior + prior
            preds.append(posterior)

        return self._classes[np.argmax(preds)]

    def _pdf(self, x, mean, var):
        num = np.exp(-((x - mean) ** 2)/ (2 * var))
        den = np.sqrt(2 * np.pi * var)

        return num/den