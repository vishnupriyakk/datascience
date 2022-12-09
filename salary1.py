# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:33:45 2022

@author: VishnuPriya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("salaries (4).csv")
label=LabelEncoder()
df['company']=label.fit_transform(df.company)
df['job']=label.fit_transform(df.job)
df['degree']=label.fit_transform(df.degree)
x=df.iloc[:,0:3]
y=df.iloc[:,-1]

model=DecisionTreeClassifier()
model.fit(x,y)
model.score(x,y)
lrmodel=LogisticRegression()
lrmodel.fit(x,y)
lrmodel.score(x,y)
