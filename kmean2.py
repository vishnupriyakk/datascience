# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:23:53 2022

@author: VishnuPriya
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
x,y=make_blobs(n_samples=400,centers=4,cluster_std=0.6,random_state=0)
plt.scatter(x[:,0],x[:,1])
plt.show()
model=KMeans(n_clusters=4)
model.fit(x)
ypred=model.predict(x)


#visualization
plt.scatter(x[:,0],x[:,1],c=ypred,s=20,cmap='summer')
centers=model.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='black',s=50)
plt.show()
