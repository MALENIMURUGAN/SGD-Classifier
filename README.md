# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Dataset: Import the Iris dataset and store it in a DataFrame.
2. Preprocess Data: Separate features (X) and target (y).
3. Split Data: Divide dataset into training and testing sets using train_test_split().
4. Initialize Model: Create an SGDClassifier with logistic regression loss function.
5. Train Model: Fit the classifier on the training data.
6. Predict: Use the trained model to predict target values for the test data.
7. Evaluate Model:The model is evaluated using accuracy score, confusion matrix and classification report.
8. Output Results: Print accuracy, confusion matrix, and classification report.
   

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Maleni M
RegisterNumber: 212223040110
*/
PROGRAM:
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris=load_iris()

df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

print(df.head())

X=df.drop('target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(X_train,y_train)

y_pred=sgd_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

```

## Output:
<img width="1578" height="1228" alt="image" src="https://github.com/user-attachments/assets/d48e4b23-79fa-4aa0-a622-2221bc2c3877" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
