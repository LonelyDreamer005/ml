import pandas as pd
import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
x = data.data[:,:2]
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state= 5)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Perceptron(max_iter=100, eta0=0.1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print('Weights', model.coef_)
print('Bias', model.intercept_)



from sklearn.datasets import load_iris 
from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score 
data = load_iris() 
X = data.data 
y = data.target 
model = Perceptron() 
model.fit(X, y) 
y_pred = model.predict(X) 
print("Accuracy:", accuracy_score(y, y_pred)) 
print("Weights:", model.coef_) 
print("Bias:", model.intercept_)