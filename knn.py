import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
data = load_iris()
x= data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state= 4)

best_k = 1
best_acc = 0

for k in range(1,11):
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(x_train, y_train)

  y_pred = model.predict(x_test)
  acc = accuracy_score(y_test, y_pred)

  if best_acc < acc:
    best_acc = acc
    best_k = k
  
final_model = KNeighborsClassifier(n_neighbors = best_k)
final_model.fit(x_test, y_test)

y_pred = final_model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f'Best K: {best_k}')
print(f'Accuracy: {acc}')