Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
1.Data Collection & Preprocessing

2.Select relevant features that impact placement

3.Import the Logistic Regression model from sklearn.

4.Train the model using the training dataset.

5.Use the trained model to predict placement for new student data.

Program:
/* Program to implement the the Logistic Regression Model to Predict the Placement Status of Student. Developed by: SALINI A RegisterNumber: 212223220091*/
'''
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
'''

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
'''
Output:
![image](https://github.com/user-attachments/assets/73e3ead7-62f6-4b4f-a343-88ff8b22b630)
![image](https://github.com/user-attachments/assets/7940ff2d-2c69-4f48-8271-d534a92d4f23)
![image](https://github.com/user-attachments/assets/251da739-6edb-4bd6-ae94-9d0f549e517e)
![image](https://github.com/user-attachments/assets/63ce9e03-0ced-41ae-82de-45325795dc4c)
![image](https://github.com/user-attachments/assets/27004e40-ce73-4974-ad87-2ad12a69f7e2)
![image](https://github.com/user-attachments/assets/af7a0d85-59c5-47dc-abda-b53e42309fa5)
![image](https://github.com/user-attachments/assets/dff5eaef-fbfd-4e9b-b8d2-81bcdeefc4bf)
![image](https://github.com/user-attachments/assets/8141ca7b-55b3-41a6-a0f5-bb8ace4cf742)
![image](https://github.com/user-attachments/assets/adb73951-cc3f-40ae-9623-846f07736787)
![image](https://github.com/user-attachments/assets/6f35737c-8d0a-4408-97b5-4c97a8f49eec)

Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
