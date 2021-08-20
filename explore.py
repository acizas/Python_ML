import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import load_data
from preprocessing import load_clean_normal_data

data = load_clean_normal_data()
#data = load_data()

#data = data.drop('Rk',axis=0)
#data.drop(data.index[0], inplace=True)
#data.drop(data.index[1], inplace=True)
#data.drop(data.index[2], inplace=True)
#data.drop(data.index[4], inplace=True)
#data = data.drop('Player',axis=1)
#data = data.drop('Pos',axis=2)    
#data = data.drop('Tm',axis=4)
                 
for stat in data.columns[3:]:
    fig,ax = plt.subplots(1,2,sharex=True,squeeze=True)
    hist, bin_edges = np.histogram(data[stat],bins=100)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2
    ax[0].bar(bin_centers,hist,width=np.diff(bin_edges))
    ax[0].set_title(stat+' Hist')
    #plt.show()
    
    cdf = np.cumsum(hist)/data.shape[0]
    ax[1].plot(bin_centers,cdf)
    ax[1].set_title(stat+' CDF')
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(data[data.columns[1]],data[data.columns[2]],data[data.columns[3]])
#ax.set_xlabel(data.columns[1])
#ax.set_xlabel(data.columns[2])
#ax.set_xlabel(data.columns[3])
#plt.show()
                
