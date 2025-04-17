# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Collection & Preprocessing

2.Select relevant features that impact placement

3.Import the Logistic Regression model from sklearn.

4.Train the model using the training dataset.

5.Use the trained model to predict placement for new student data. 

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SALINI A
RegisterNumber:  212223220091
*/

```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
```
data1.isnull().sum()
```
```
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
```
```
x=data1.iloc[:,:-1]
x
```
```
y=data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

![image](https://github.com/user-attachments/assets/6318e917-0cb5-4975-8871-74023e5298b2)


![image](https://github.com/user-attachments/assets/4b961ce6-7695-4efd-9c66-1396f96995db)


![image](https://github.com/user-attachments/assets/04aba851-7b21-46f2-a5ef-dbe316fc6e01)


![image](https://github.com/user-attachments/assets/1dd523e5-915d-4c15-b2e4-d56427d556d6)


![image](https://github.com/user-attachments/assets/7736e098-8819-41ef-8422-854c4db6d828)


![image](https://github.com/user-attachments/assets/dd537b87-a67d-4db2-827e-c4f45e83a4d8)


![image](https://github.com/user-attachments/assets/a42f3d95-539a-4f85-9051-163bd80d30e7)


![image](https://github.com/user-attachments/assets/4f75dc30-f501-4795-9281-dcdb5da767bc)


![image](https://github.com/user-attachments/assets/77df2edd-195f-41b2-a558-0b97212365b2)


![image](https://github.com/user-attachments/assets/75cfc73e-6ce4-4298-acc3-e6cbc2fd20e0)


![image](https://github.com/user-attachments/assets/18fbd5f7-90dc-48dd-ba23-705d1de27381)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
