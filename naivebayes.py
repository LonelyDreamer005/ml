import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
x = data.data
y = data.target 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)