from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
df=load_breast_cancer()
x=df.data
y=df.target
print(df.feature_names)
x.shape
y.shape
model=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)
model.fit(x,y)
ypred=model.predict(x)
ypred
cm=confusion_matrix(y,ypred)
ac=accuracy_score(y,ypred)
cr=classification_report(y,ypred)
