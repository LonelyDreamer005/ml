import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('/content/sample_data/Iris1.csv')

x = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = data['Species']

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state= 5)

model = DecisionTreeClassifier(criterion='gini')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred) * 100)

plt.figure(figsize=(20,10))
plot_tree(model,
          filled=True,
          feature_names=x.columns,
          class_names=le.classes_)
plt.show()

