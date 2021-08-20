import csv
import numpy as np

with open('white_wines.csv','r') as infile:
    reader = csv.reader(infile)
    columnNames = next(reader)
    dat = np.array(list(reader),dtype=float)

labelIndex = columnNames.index('Outcome')
labels = dat[:,labelIndex]

import matplotlib.pyplot as plt

plt.hist(labels)
plt.show()
