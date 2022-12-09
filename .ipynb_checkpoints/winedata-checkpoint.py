import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_csv("wine.csv")
df.isnull().sum()
x=df.iloc[:,0:13].values
y=df.iloc[:,-1].values
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
model.score(xtest,ytest)
