import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
df = pd.read_csv('archive (2).zip')
df
df.isnull().sum()
x=df.iloc[:,0:8].values
x
y=df.iloc[:,8:9].values
y
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=0)
xtrain
xtest
ytrain
ytest
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(ytest,ypred)
cm
cr=classification_report(ytest,ypred)
cr
ac=accuracy_score(ytest,ypred)
ac
model.score(xtest,ytest)

