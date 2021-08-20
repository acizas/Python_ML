from preprocessing import load_data, separate labels

columns, data = load_data("diabetes.csv")
columnNames, xs, ys = separate_labels(columns,data)

import matplotlib.pyplot as plt

feature1 = 1
feature2 = 2

newfeature = xs[:,feature2]<1.0
print(newfeature)
print(xs.shape)
print(newfeature.shape)
xs = np.stack((xs,newfeature),axis=1)
print(xs)

plt.scatter(xs[:,ature1],xs[:,feature2],ys,cmap=jet)
plt.xlabel(columnNames[feature1])
plt.ylabel(columnNames[feature2])
plt.show
