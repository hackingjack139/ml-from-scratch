import numpy as np
from naive_bayes import naive_bayes
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 1)

classifier = naive_bayes()
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

accuracy = np.sum(preds == y_test) / len(y_test)
print(accuracy)