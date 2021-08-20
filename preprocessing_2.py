import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import csv

def load_data(filename='white_wine.csv'):
    with open(filename,'r') as file:
        reader = csv.reader(file)
        columnNames = next(reader)
        rows = np.array(list(reader), dtype=float)
    return columnNames, rows


#def joke_load_data(filename='diabetes.csv'):
#    print("Nah, I won't load ",filename," today.")
#    print("I print out only once!")

#joke_load_data('joke.csv')

def separate_labels(columnNames, rows):
    labelColumnIndex = columnNames.index('quality')
    ys = rows[:, labelColumnIndex]
    xs = np.delete(rows,labelColumnIndex,axis=1)
    del columnNames[labelColumnIndex]
    return columnNames, xs, ys


