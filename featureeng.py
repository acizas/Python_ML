from preprocessing import load_data, separate labels

columns, data = load_data("white_wine.csv")
columnNames, x, y = separate_labels(columns,data)

import matplotlib.pyplot as plt

feature1 = 1
feature2 = 2

newfeature = x[:,feature2]<1.0
print(newfeature)
print(x.shape)
print(newfeature.shape)
x = np.stack((xs,newfeature),axis=1)
print(x)

plt.scatter(x[:,ature1],xs[:,feature2],y,cmap=jet)
plt.xlabel(columnNames[feature1])
plt.ylabel(columnNames[feature2])
plt.show