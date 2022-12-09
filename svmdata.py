# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 19:50:03 2022

@author: VishnuPriya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
data=pd.read_csv("archive.zip")
x=data.iloc[:,0:2]
y=data.iloc[:,-1]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=(0))
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
model=SVC(kernel='linear',random_state=0)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
cm
cr=classification_report(ytest,ypred)
ac=accuracy_score(ytest,ypred)
#visualization
from matplotlib.colors import ListedColormap  
x_set, y_set = xtrain, ytrain  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
    c = ListedColormap(('purple', 'green'))(i), label = j)  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  