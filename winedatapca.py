# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:26:12 2022

@author: VishnuPriya
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

df=pd.read_csv("Wine.csv")

x=df.iloc[:,0:13].values
y=df.iloc[:,-1].values
sc=StandardScaler()
X=sc.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)
model=RandomForestClassifier(n_estimators=10,criterion='entropy')
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)

#acc=97