from sklearn.datasets import load_iris
from knn import KNN
from sklearn.model_selection import train_test_split
import numpy as np

iris_pd = load_iris()
X = iris_pd.data
y = iris_pd.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

classifier = KNN(k=10)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(accuracy)