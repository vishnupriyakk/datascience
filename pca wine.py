# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:55:56 2022

@author: VishnuPriya
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

df=pd.read_csv("Wine.csv")
df.corr()["Customer_Segment"].sort_values()
x=df.iloc[:,0:13].values
y=df.iloc[:,-1].values
sc=StandardScaler()
X=sc.fit_transform(x)


#applying PCA on featuers      PCA=principle componenet analysis-used for dimentionality reduction
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x1=pca.fit_transform(X)
explained_variance=pca.explained_variance_ratio_
print(sum(explained_variance))
xtrain,xtest,ytrain,ytest=train_test_split(x1,y,train_size=0.8,random_state=0)
model=RandomForestClassifier(n_estimators=10,criterion='entropy')
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
#acc=97