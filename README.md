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
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: salini a
RegisterNumber:  21223220091
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

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

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

![image](https://github.com/user-attachments/assets/6e2948e9-a001-47b3-b7d1-0a3167f9bc12)
![image](https://github.com/user-attachments/assets/c589784e-1a27-43e5-88de-04d4c72bb3ef)
![image](https://github.com/user-attachments/assets/d9a6e27b-927c-4597-90f3-7f4aa0849a4d)
![image](https://github.com/user-attachments/assets/992133a4-b634-4086-8af2-2c12d27a9a30)
![image](https://github.com/user-attachments/assets/a6afb8ff-5721-4e30-9368-84d1588de374)
![image](https://github.com/user-attachments/assets/4da6e4eb-df75-4152-aaf0-d4a56fbcba0c)

![image](https://github.com/user-attachments/assets/a712cc27-f6c7-4eb5-bb66-d5859a2ad1da)
![image](https://github.com/user-attachments/assets/a601baf5-1953-4fef-89b9-d017b36d5497)
![image](https://github.com/user-attachments/assets/b825b19b-b4ba-4bbc-afa5-e2d0a2fd1aa0)
![image](https://github.com/user-attachments/assets/22595679-43bb-4d31-9a20-d0a6facd0742)
![image](https://github.com/user-attachments/assets/56ec0ae4-a097-43ce-986d-eb19cf3365df)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
