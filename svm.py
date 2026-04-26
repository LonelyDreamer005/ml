import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = datasets.load_breast_cancer()

x = data.data[:,:2]
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state= 5)

linear_model = SVC(kernel='linear')
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_test)

rbf_model = SVC(kernel='rbf')
rbf_model.fit(x_train, y_train)
y_pred_rbf = rbf_model.predict(x_test)

print(accuracy_score(y_test, y_pred_linear))
print(accuracy_score(y_test, y_pred_rbf))
def plot_bon(model, x,y,title):
  x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
  y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
  z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  z = z.reshape(xx.shape)
  plt.figure()
  plt.contourf(xx, yy, z, alpha=0.4)
  plt.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor='k')
  plt.title(title)
  plt.show()

plot_bon(linear_model, x,y, 'Linear Kernel')
plot_bon(rbf_model, x,y, 'RBF Kernel')

