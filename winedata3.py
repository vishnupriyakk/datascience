# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:50:15 2022

@author: VishnuPriya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
wine=load_wine()
wine
dir(wine)
wine['data']
wine['feature_names']
wine['target']
wine['target_names']
df=pd.DataFrame(wine["data"],columns=wine["feature_names"])
x=wine.data
y=wine.target
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=0)
model=GaussianNB()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
model.score(xtrain,ytrain)
model1=BernoulliNB()
model1.fit(xtrain,ytrain)
ypred=model1.predict(xtest)
model1.score(xtrain,ytrain)
model2=MultinomialNB()
model2.fit(xtrain,ytrain)
ypred=model2.predict(xtest)
model2.score(xtrain,ytrain)
