# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:48:12 2022

@author: VishnuPriya
"""

import pandas as pd
import pickle
f1=open(file="naivemodel.pkl",mode="br")
m1=pickle.load(f1)
f1.close()
f2=open(file="standrdmodel.pkl",mode="br")
sc1=pickle.load(f2)
f2.close()
def prediction(a,b):
    data={'age':a,'estimatedsalary':b}
    df=pd.DataFrame(data,index=[0])
    df=sc1.transform(df)
    pred=m1.predict(df)
    if int(pred)==1:
        return 'purchased'
    else:
        return 'not purchased'
age=int(input("enter your age"))
salary=int(input("enter your salary"))
prediction(age,salary)   
