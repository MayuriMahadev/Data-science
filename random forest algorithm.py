# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:53:59 2024

@author: Admin
"""

#####################12/11/2024###################
import pandas as pd
df=pd.read_csv("C:/Dataset/movies_classification.csv")
df.info()


df=pd.get_dummies(df,columns=["3D_available","Genre"],drop_first=True)

predictors=df.loc[:,df.columns!="Start_Tech_Oscar"]


target=df["Start_Tech_Oscar"]
################################
# let us partition the data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2)
###################################
##model selection
from sklearn.ensemble import RandomForestClassifier 
rand_for=RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=42)
# n_estimate ;it is number of train in the forest always in the range 500
# to 1000
# n_jobs=1 means no of jobs running parrallel =1
# if it is -1 then multiple jobs running parrale
# random_state=controls randoness in bootstapping
# boostrap applying getting samples replaced
rand_for.fit(X_train,y_train)
pred_X_train=rand_for.predict(X_train)
pred_X_test=rand_for.predict(X_test)
##############
# let us check the performance of the model
from sklearn.metrics import accuracy_score,confusion_matrix 
accuracy_score(pred_X_test,y_test)
confusion_matrix(pred_X_test,y_test)

