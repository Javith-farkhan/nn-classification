# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![NN_MODEL_2](https://user-images.githubusercontent.com/94296805/228287390-749f9a75-5a35-468b-961c-b4d4d52e3124.png)


## DESIGN STEPS

### STEP 1:

Load the csv file and then use the preprocessing steps to clean the data

### STEP 2:

Split the data to training and testing

### STEP 3:

Train the data and then predict using Tensorflow


## PROGRAM
c#

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

import pandas as pd
customer_df=pd.read_csv('customers.csv')

customer_df.head(10)

customer_df.columns
customer_df.dtypes
customer_df.shape
customer_df.isnull().sum()
customer_df_cleaned=customer_df.dropna(axis=0)
customer_df_cleaned.shape
customer_df_cleaned['Gender'].unique()
customer_df_cleaned['Ever_Married'].unique()
customer_df_cleaned['Graduated'].unique()
customer_df_cleaned['Profession'].unique()
customer_df_cleaned['Spending_Score'].unique()
customer_df_cleaned['Var_1'].unique()
customer_df_cleaned['Segmentation'].unique()


categories_list=[['Male', 'Female'],['No', 'Yes'],['No', 'Yes'],['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor','Homemaker', 'Entertainment', 'Marketing', 'Executive'],['Low', 'High', 'Average'],]
enc=OrdinalEncoder(categories=categories_list)


customers_1 = customer_df_cleaned.copy()

customers_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']] = enc.fit_transform(customers_1[['Gender', 'Ever_Married','Graduated','Profession','Spending_Score']])

customers_1.dtypes

le=LabelEncoder()

customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])

customers_1.dtypes

customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)

customers_1.dtypes


corr = customers_1.corr()

corr = customers_1.corr()

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
sns.pairplot(customers_1)
sns.distplot(customers_1['Age'])     
plt.figure(figsize=(10,6))
sns.countplot(customers_1['Family_Size'])

X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1 = customers_1[['Segmentation']].values

y1[10]

one_hot_enc = OneHotEncoder()

one_hot_enc.fit(y1)
y1.shape

y = one_hot_enc.transform(y1).toarray()

y.shape

y1[0]

y[0]

X.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=50)
     

X_train[0]

X_train.shape

scaler_age = MinMaxScaler()

scaler_age.fit(X_train[:,2].reshape(-1,1))

X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

ai_brain =Sequential([Dense(8,input_shape = (8,) ),
    Dense(32,activation="relu"),
    Dense(4,activation='softmax')
])

ai_brain.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

ai_brain.fit(x=X_train_scaled,y=y_train,epochs=2000,batch_size=256,validation_data=(X_test_scaled,y_test),)

metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)

x_test_predictions.shape

y_test_truevalue = np.argmax(y_test,axis=1)

y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))

x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1) 

print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))
             
                                                                                                                               
                                                               
## Dataset Information

![Dataset_2](https://user-images.githubusercontent.com/94296805/228288382-7b629678-364c-4583-b44d-7d777a7895fc.png)


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Loss_graph](https://user-images.githubusercontent.com/94296805/228288528-77641320-4226-40a5-a028-80ab284976d8.png)


### Classification Report

![Classification_report](https://user-images.githubusercontent.com/94296805/228288684-46c56eab-2190-462b-9c90-fc5feffaa430.png)


### Confusion Matrix


![Confusion_matrix](https://user-images.githubusercontent.com/94296805/228288752-3e2e67fd-8401-4937-8a7f-b077b3d3ab29.png)


### New Sample Data Prediction


![New_prediction](https://user-images.githubusercontent.com/94296805/228289560-351bcbd8-4211-4a74-9319-4c1e9dda9aca.png)


## RESULT

Thus a Neural Network Classification Model is created and executed successfully
