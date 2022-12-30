# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:56:15 2022

@author: VishnuPriya
"""

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers.csv")
x=data.iloc[:,3:5].values
plt.scatter(x[:,0],x[:,1])
plt.show()
#applying k-mean clustering
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    model=KMeans(n_clusters=i)
    model.fit(x)
    a=model.inertia_
    wcss.append(a)#wcss is sum of squares/to calculate the varience
plt.plot(1,11,wcss)
plt.xlabel("cluster")
plt.ylabel("wcss")
plt.title("finding clusters count")
plt.show()


model1=KMeans(n_clusters=5)
model1.fit(x)
y=model1.predict(x)
print(set(y))#clusterne distict value aakkan
#visualization of clusterd data
plt.scatter(x[y==0,0],x[y==0,1],color='blue',label="first cluster")
plt.scatter(x[y==1,0],x[y==1,1],color='green',label="second cluster")
plt.scatter(x[y==2,0],x[y==2,1],color='yellow',label="third cluster")
plt.scatter(x[y==3,0],x[y==3,1],color='purple',label="four cluster")
plt.scatter(x[y==4,0],x[y==4,1],color='skyblue',label="five cluster")
plt.legend()
#centeroid
plt.scatter(model1.cluster_centers_[:,0],model1.cluster_centers_[:,1],c="red",s=100)
plt.show()

