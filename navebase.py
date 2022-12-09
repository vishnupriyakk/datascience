# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:25:39 2022

@author: VishnuPriya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
data=pd.read_csv("archive.zip")
x=data.iloc[:,0:2]
y=data.iloc[:,-1]
sc=StandardScaler()
X=sc.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)
model=GaussianNB()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
cr=classification_report(ytest,ypred)

#model saving
import pickle
f1=open(file="naivemodel.pkl",mode="bw")
pickle.dump(model,f1)        #modelile data f1lekk dump cheyuka
f1.close()
f2=open(file="standrdmodel.pkl",mode="bw")
pickle.dump(sc,f2)
f2.close()