import numpy as np
import csv
import pandas
import preprocessing
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

columnNames, rows = preprocessing.load_data('nba_players_stats_19_20_per_game.csv')
columnNames, data = preprocessing.load_data('nba_players_stats_19_20_per_game.csv')


columnNames, x, y = preprocessing.separate_labels(columnNames, data)
columnNames, z = preprocessing.load_data('unknowns.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train
y_train
x_test
y_test
train_test_split(y, shuffle=True)

x, y = load_iris(return_X_y=True)

clf = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(100,), random_state=5).fit(x_train,y_train)

print(clf.score(x_test, y_test))

pred = clf.predict(z)

df = pandas.DataFrame(data)
df.to_csv("test.csv")
print(df.to_csv())
