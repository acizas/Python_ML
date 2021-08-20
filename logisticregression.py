import numpy as np
import csv
import pandas
import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

columnNames, rows = preprocessing.load_data('white_wine.csv')
columnNames, data = preprocessing.load_data('white_wine.csv')

columnNames, x, y = preprocessing.separate_labels(columnNames, data)
columnNames, z = preprocessing.load_data('unknowns.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
x_train
y_train
x_test
y_test
train_test_split(y, shuffle=False)

x, y = load_iris(return_X_y=True)

clf = LogisticRegression(penalty='l2', random_state=5, solver='liblinear').fit(x_train,y_train)

print(clf.score(x_test, y_test))

pred = clf.predict(z)

df = pandas.DataFrame(pred)
df.to_csv("scores.csv")
print(df.to_csv())
