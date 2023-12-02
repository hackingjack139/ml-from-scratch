from logistic_regression import logistic_regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

classifier = logistic_regression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]

accuracy = np.sum(y_pred == y_test)/len(y_test)
print(accuracy)