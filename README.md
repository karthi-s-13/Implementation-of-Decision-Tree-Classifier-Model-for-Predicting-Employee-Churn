# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check number of unqiue element present in column 'left' by using value_counts() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KARTHIKEYAN S
RegisterNumber:  212224230116
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
```

```
data = pd.read_csv('Employee.csv')
df = pd.DataFrame(data)
df.head()
```
```
df['left'].value_counts().reset_index()
```
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])
df.head()
```
```
x = df.drop(['left','Departments '], axis=1)
x.head()
```
```
y = df['left']
y.head()
```
```
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
```
```
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
```
```
predict = dt.predict(x_test)
predict
```
```
accuracy = accuracy_score(y_test,predict)
print(f"Accuracy:{accuracy}")
```


## Output:
### Dataset
![Screenshot 2025-05-15 083320](https://github.com/user-attachments/assets/f0c25633-7576-4d6d-a2e7-3d1420a7eced)

### After LabelEncoding
![Screenshot 2025-05-15 083400](https://github.com/user-attachments/assets/a94fb3c2-1769-4854-942a-e9c647eca8f2)

### X dataset
![Screenshot 2025-05-15 083410](https://github.com/user-attachments/assets/cdbcb70c-9017-4b3c-a6cb-3fce583515f1)

### Y dataset
![Screenshot 2025-05-15 083415](https://github.com/user-attachments/assets/b31855c0-3a0c-4d81-a5a0-9ab642e24e16)

### Predicted Value
![Screenshot 2025-05-15 083423](https://github.com/user-attachments/assets/e1a88e6e-6bea-4a71-a805-87c458fb37d5)

### Accuracy
![Screenshot 2025-05-15 083429](https://github.com/user-attachments/assets/42c8c24e-5a10-41d7-99b9-2b873bd447cf)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
