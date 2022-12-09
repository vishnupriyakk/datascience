# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:38:32 2022

@author: VishnuPriya
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
data=pd.read_csv("archive.zip")
x=data.iloc[:,0:2]
y=data.iloc[:,-1]
sc=StandardScaler()
X=sc.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)
model=DecisionTreeClassifier(criterion="entropy",random_state=0)
model.fit(xtrain,ytrain)
tree=export_text(model,feature_names=["age","salary"])
ypred=model.predict(xtest)
cm=confusion_matrix(ytest,ypred)
ac=accuracy_score(ytest,ypred)
cr=classification_report(ytest,ypred)
lrmodel=LogisticRegression()
lrmodel.fit(xtrain,ytrain)
ypred1=lrmodel.predict(xtest)
cm1=confusion_matrix(ytest,ypred1)
ac1=accuracy_score(ytest,ypred1)
knmodel=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)
knmodel.fit(xtrain,ytrain)
ypred2=knmodel.predict(xtest)
cm2=confusion_matrix(ytest,ypred2)
ac2=accuracy_score(ytest,ypred2)
svmodel=SVC(kernel='linear',random_state=0)
svmodel.fit(xtrain,ytrain)
ypred3=svmodel.predict(xtest)
model4=GaussianNB()
model4.fit(xtrain,ytrain)
ypred4=model4.predict(xtest)
#roc,auc curve

#from sklearn.metrics import roc_curve,auc,roc_auc_score
#fpr,tpr,thresh=roc_curve(ytest,ypred)


#a=auc(fpr,tpr)
#plt.plot(fpr,tpr,color="green",label=("AUC value: %0.2f"%(a)))
#plt.plot([0,1],[0,1],"--",color="red")
#plt.xlabel("False positive rate")
#plt.ylabel("True positive rate")
#plt.title("ROC-AUC CURVE")
#plt.legend(loc="best")
#plt.show()

from sklearn.metrics import roc_auc_score,roc_curve,auc
fpr,tpr,thresh=roc_curve(ytest,ypred)
a = auc(fpr,tpr)


fpr1,tpr1,thresh = roc_curve(ytest,ypred1)
b = auc(fpr1,tpr1)


fpr2,tpr2,thresh=roc_curve(ytest, ypred2)
c=auc(fpr2,tpr2)


fpr3,tpr3,thresh=roc_curve(ytest,ypred3)
d=auc(fpr3,tpr3)


fpr4,tpr4,thresh=roc_curve(ytest,ypred4)
e=auc(fpr4,tpr4)



plt.plot(fpr,tpr,color="green",label=("AUC value of Decision tree: %0.2f"%(a)))
plt.plot(fpr1,tpr1,color="blue",label=("AUC value of logistic Regression: %0.2f"%(b)))
plt.plot(fpr2,tpr2,color="yellow",label=("AUC value of knn: %0.2f"%(c)))
plt.plot(fpr3,tpr3,color="red",label=("AUC value of svm: %0.2f"%(d)))
plt.plot(fpr4,tpr4,color="purple",label=("AUC value of naivebayes: %0.2f"%(e)))
plt.plot([0,1],[0,1],"--",color="red")
plt.xlabel("False positive rate")
plt.ylabel("True Positive rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="best")
plt.show()


