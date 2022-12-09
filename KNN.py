import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("archive.zip")
df.isnull().sum()
x=df.iloc[:,[0,1]].values
y=df.iloc[:,2].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.8,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
#acc=[]
knmodel=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)    #p=neuclidian matric
knmodel.fit(xtrain,ytrain)
#predicting the test result
ypred=knmodel.predict(xtest)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(ytest,ypred)
cm
cr=classification_report(ytest,ypred)
cr
ac=accuracy_score(ytest,ypred)
ac
#for i in range(1,11):
  # knmodel=KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)
  # #knmodel.fit(xtrain,ytrain)    
  # pred=knmodel.predict(xtest)
  # a=accuracy_score(ytest,pred) 
  # acc.append(a)
#plt.plot(acc)
#visualizing of training data
from matplotlib.colors import ListedColormap  
x_set, y_set = xtrain, ytrain  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, knmodel.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
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

#visualizing of testing data
from matplotlib.colors import ListedColormap  
x_set, y_set = xtest, ytest  
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(x1, x2, knmodel.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
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
