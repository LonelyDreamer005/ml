[Short GPT link](https://chatgpt.com/s/t_69ee3f56c38c81918e091b45b825590f)

[Colab Link](https://colab.research.google.com/drive/1D79sx4dEgI25swCvscWKR_keguukeDKQ?usp=sharing)

[Manikanta's git ->](https://github.com/Manikanta6205/ml)


from sklearn.datasets import load_iris

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import Perceptron

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture



# Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)

ypred = model.predict(x)
```

---

# Linear Regression Without Library

```python
m = slope_formula
c = intercept_formula

ypred = [m*i + c for i in x]
```

---

# Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x, y)

yprob = model.predict_proba(x)
```

---

# KNN

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(xtr, ytr)

ypred = model.predict(xte)
```

---

# Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(xtr, ytr)

ypred = model.predict(xte)
```

---

# Perceptron

```python
from sklearn.linear_model import Perceptron

model = Perceptron()

model.fit(x, y)

ypred = model.predict(x)
```

---

# Decision Tree (CART)

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion="gini")

model.fit(xtr, ytr)
```

---

# Decision Tree (ID3)

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion="entropy")

model.fit(xtr, ytr)
```

---

# Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model.fit(xtr, ytr)
```

---

# Bagging

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50
)

model.fit(xtr, ytr)
```

---

# AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=50)

model.fit(xtr, ytr)
```

---

# XGBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(xtr, ytr)
```

---

# KMeans Clustering

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

model.fit(x)
```

---

# Gaussian Mixture Model (EM)

```python
from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components=3)

model.fit(x)
```

---

# Linear SVM

```python
from sklearn.svm import SVC

model = SVC(kernel="linear")

model.fit(x, y)
```

---

# Non Linear SVM

```python
from sklearn.svm import SVC

model = SVC(kernel="rbf")

model.fit(x, y)
```

---

# Accuracy Score

```python
from sklearn.metrics import accuracy_score

print(accuracy_score(yte, ypred))
```

---

# Train Test Split

```python
from sklearn.model_selection import train_test_split

xtr, xte, ytr, yte = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)
```

---

# Iris Dataset

```python
from sklearn.datasets import load_iris

data = load_iris()

x = data.data
y = data.target
```

---

# Breast Cancer Dataset

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = data.data
y = data.target
```

---

# Common ML Flow

```python
from sklearn...

# Load dataset
x, y = ...

# Split data
xtr, xte, ytr, yte = train_test_split(...)

# Create model
model = ModelName()

# Train model
model.fit(xtr, ytr)

# Predict
ypred = model.predict(xte)

# Accuracy
print(model.score(xte, yte))
```