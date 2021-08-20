import numpy as np
import pandas as pd
from preprocessing import load_clean_normal_data, selected_features
from models import train_one_class_svm, train_elliptic_envelope, train_isolation_forest
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data = load_clean_normal_data()

#clf = train_one_class_svm(data)
clf = train_elliptic_envelope(data)
#clf = train_isolation_forest(data)

scores = clf.decision_function(data[selected_features])

topthreeIndices = np.argsort(scores)[:3]
topthree = data.iloc[topthreeIndices]

outliers = scores < 0
outliersSansTopThree = outliers.copy()
outliersSansTopThree[topthreeIndices] = False

import matplotlib.pyplot as plt

fig, ax = plt.subplots(3,3,sharex=True, squeeze=False)

#for rownum, (_,obj) in enumerate(topthree.iterrows()):
#    for colnum, stat in enumerate(selected_features):
#        fig,ax = plt.subplots(1,2,sharex=True,squeeze=True)
#        hist, bin_edges = np.histogram(data[stat],bins=100)
#        bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
#        ax[0].bar(bin_centers,hist,width=np.diff(bin_edges))
#        ax[0].set_title(stat+' Hist')
        #plt.show()
    
#        cdf = np.cumsum(hist)/data.shape[0]
#        ax[1].plot(bin_centers,cdf)
#        ax[1].set_title(stat+' CDF')
#        plt.show()
    

for rownum, (_,obj) in enumerate(topthree.iterrows()):
    for colnum, stat in enumerate(selected_features):
        hist, bin_edges = np.histogram(data[stat],bins=100)
        cdf = np.cumsum(hist)/data.shape[0]
        bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
        ax[rownum,colnum].plot(bin_centers,cdf)
        ax[rownum,colnum].set_title(obj['Player']+' on '+stat)
        ax[rownum,colnum].plot([obj[stat],obj[stat]],[0.0,1.0])
        #ax.set_xlabel('Occurences')
        #ax.set_ylabel('Probability')
        ax[rownum,colnum].set_xlabel('Occurences')
        ax[rownum,colnum].set_ylabel('Probability')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#ax.set_title(stat)
ax.scatter(data.iloc[~outliers][selected_features[0]],
           data.iloc[~outliers][selected_features[1]],
           data.iloc[~outliers][selected_features[2]])
ax.scatter(data.iloc[outliersSansTopThree][selected_features[0]],
           data.iloc[outliersSansTopThree][selected_features[1]],
           data.iloc[outliersSansTopThree][selected_features[2]])
ax.scatter(topthree[selected_features[0]],
           topthree[selected_features[1]],
           topthree[selected_features[2]])

#plt.set_xlabel('3P')
#plt.set_ylabel('FGA')
#plt.set_zlabel('FT')
plt.show()
                        
    
