x = [1,2,3,4,5,6]
y = [2,4,5,6,3,6]
n = len(x)
sx = sum(x)
sy = sum(y)
xy = sum(x[i]*y[i] for i in range(n))
x2 = sum(x[i]**2 for i in range(n))
m = (n*xy - (sx*sy))/(n*x2 - sx**2)
c = (sy - m*sx)/n
# print(m, n)

def predic(o):
  return m*o + c

y_pred = [predic(o) for o in x]
mse = sum((y[i]-y_pred[i])**2 for i in range(n)) / n
rmse = math.sqrt(mse)

plt.figure()
plt.scatter(x,y, label = 'Actual')
plt.plot(x,y_pred, label = 'Line')
x_test = 5
y_test = predic(x_test)
plt.scatter(x_test, y_test, label = 'Test')
plt.legend()
plt.grid()
plt.show()

