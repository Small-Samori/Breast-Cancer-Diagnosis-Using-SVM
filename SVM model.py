import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors, svm

# reading in dataset and doing preprocessing
df = pd.read_csv('breast-cancer-dataset.txt')
df.replace("?", -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# splitting data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# defining the classifer
clf = svm.SVC(gamma='auto')
clf.fit(X_train, y_train)

# evaluating model
accuracy = clf.score(X_test, y_test)
print(f'Accuracy: {accuracy}\n')

# a random prediction
example_measure = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measure = example_measure.reshape(len(example_measure), -1)

prediction = clf.predict(example_measure)
print(f'Prediction: {prediction}')
