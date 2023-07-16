import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lr = LinearRegression()
lr.fit(X, y)

pr = PolynomialFeatures(degree=4)
X_poly = pr.fit_transform(X)

lr_2 = LinearRegression()
lr_2.fit(X_poly, y)

y_pre = lr.predict(X)
y_poly_pre = lr_2.predict(X_poly)

plt.scatter(X, y)
plt.plot(X, y_poly_pre, color="red")
plt.plot(X, y_pre, color="blue")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

print(np.sqrt(mean_squared_error(y, y_pre)))
print(r2_score(y, y_pre))

print(np.sqrt(mean_squared_error(y, y_poly_pre)))
print(r2_score(y, y_poly_pre))

X_poly_test = pr.fit_transform([[4.5], [5.5]])
