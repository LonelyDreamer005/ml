import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('/content/sample_data/Iris1.csv')

x = data.iloc[:,:-1]
y = data.iloc[:,-1]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state= 5)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

best_k = 1
best_acc = 0
for k in range(1,11):
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(x_train, y_train)

  acc = accuracy_score(y_test, model.predict(x_test))
  if acc > best_acc:
    best_acc = acc
    best_k = k
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(x_train, y_train)

print(best_k, accuracy_score(y_test, final_model.predict(x_test)))

