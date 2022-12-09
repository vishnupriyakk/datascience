import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as nm
import matplotlib.pyplot as mtp
df=pd.read_csv("archive.zip")

df.shape
df.isnull()
df.isnull().sum()
x=df.iloc[:,[0,1]].values

y=df.iloc[:,2].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm  

from matplotlib.colors import ListedColormap  
x_set, y_set = xtrain, ytrain  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, model.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
    c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Logistic Regression (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()  
