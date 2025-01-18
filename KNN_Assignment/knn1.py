# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:05:22 2025

@author: Admin
"""


Business Problem:
The company aims to classify animals into different species based on their physical features. Accurate classification can help in animal conservation efforts, wildlife monitoring, and ensuring the health and safety of different species by identifying them correctly in various settings such as zoos or wildlife sanctuaries.

Business Objective:
The main objective of this model is to build a predictive model using KNN to categorize animals into different species based on their features. By achieving this, the company can:

Enhance Animal Classification: Help zoologists and wildlife experts classify animals more efficiently based on physical features.
Support Conservation Efforts: Improve the identification and tracking of endangered species, aiding in conservation and protection initiatives.
Facilitate Wildlife Monitoring: Assist wildlife authorities in identifying and monitoring animal populations in both captivity and the wild, enabling more effective management and care.
Improve Animal Healthcare: Identifying species accurately helps in providing tailored healthcare and ensuring proper treatment for different animals.
Constraints:
Data Quality: Missing or inaccurate data on animal features could impact the classification accuracy, especially if certain features are inconsistent or unrecorded.
Model Interpretability: Although KNN is effective, it may not always provide insights into why specific animals are classified a certain way. The model should provide enough transparency for decision-makers in wildlife management.
Scalability: As the dataset of animals expands, the KNN model might face performance issues, especially with a large number of features or animals.
Adaptability: The model should be adaptable to classify new species based on incoming data, which may require regular updates.
Bias Reduction: The model must ensure that no particular animal group is misclassified due to biases in the dataset, ensuring fairness and accuracy across species.

# Importing necessary libraries
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
email_data = pd.read_csv("E:/data_science_revision/Assignments/K-Nearest Neighbors/ZooKNN/Zoo.csv.xls", encoding="ISO-8859-1")
email_data
# Data cleaning
email_data.fillna(value='missing', inplace=True)

# EDA
email_data.info()
email_data.describe()

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(email_data['Na'], kde=True)
plt.title('Histogram of Na')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Na', data=email_data)
plt.title('Boxplot of Na')
plt.xlabel('Age')
plt.ylabel('Values')
plt.show()

#Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(email_data['Na'])
plt.title('scatterplot of Na')
plt.xlabel('Na')
plt.ylabel('Values')
plt.show()

# Model Building

email_data.describe()
email_data['Na'] = np.where(email_data['Na'] == 'B', 'Beniegn', email_data['Na'])
email_data['Na'] = np.where(email_data['Na'] == 'M', 'Malignant', email_data['Na'])
########################################################################33 
#0th Column is patient ID let us drop it 
email_data = email_data.iloc[:,1:32]
#########################################################################3
#Normalisation 
def norm_func(i):
    return (i-i.min())/(i.max()-i.min())
email_data_n = norm_func(email_data.iloc[:,1:32])
#becaue now 0th column is output or lable it is not considered hence l
####################################################################### 
X = np.array(email_data_n.iloc[:,:])
y = np.array(email_data['Na'])
##############################################################33 
#Now let us split the data into training and testing 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred
#Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y_test))
pd.crosstab(pred, y_test)

############################################################## 
#le tus try to select correct value of k 
acc=[]
#Running KNN algo for k=3 to 50 in the step of 2
#k value selected is odd value 
for i in range(3,50,2):
    #Declare the model 
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    acc.append([train_acc,test_acc])

import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0]for i in acc],'ro-')
plt.plot(np.arange(3,50,2),[i[0]for i in acc],'bo-')

#There are 3, 5,7, and 9 possible values where accuracy is goot
#let us check for k=3
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

accuracy_score(pred, y_test)
pd.crosstab(pred, y_test)

######==================EDA=================####
