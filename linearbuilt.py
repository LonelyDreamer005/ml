import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = np.array([1,2,4,8]).reshape(-1,1)
y = np.array([4,5,6,7])

model = LinearRegression()
model.fit(x,y)

m = model.coef_[0]
c = model.intercept_
print("m",m, "c", c)

y_pred = model.predict(x)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

x_test = np.array([5])
y_test = model.predict(x_test.reshape(-1,1))

plt.figure()
plt.scatter(x,y, label = 'Actual')
plt.plot(x,y_pred, label = 'Line')
plt.scatter(x_test, y_test, label = 'Test')
plt.legend()
plt.grid()
plt.show()


