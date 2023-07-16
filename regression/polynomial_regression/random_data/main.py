import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import operator

np.random.seed(0)

X = 2 - 3 * np.random.normal(0, 1, 20)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 20)

X = X[:, np.newaxis]

lr = LinearRegression()
lr.fit(X, y)

print(f"RMSE of original linear regression is {np.sqrt(mean_squared_error(y, lr.predict(X)))}")
print(f"R2 score of original linear regression is {r2_score(y, lr.predict(X))}")

pf = PolynomialFeatures(degree=4)
X_poly = pf.fit_transform(X)

lr_poly = LinearRegression()
lr_poly.fit(X_poly, y)
y_poly_pred = lr_poly.predict(X_poly)

print(f"RMSE of poly linear regression is {np.sqrt(mean_squared_error(y, y_poly_pred))}")
print(f"R2 score of poly linear regression is {r2_score(y, y_poly_pred)}")

plt.scatter(X, y, color="blue")
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
X, y_poly_pred = zip(*sorted_zip)
plt.plot(X, lr_poly.predict(pf.fit_transform(X)), color="red")
plt.show()