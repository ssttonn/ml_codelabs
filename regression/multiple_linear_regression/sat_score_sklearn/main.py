import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import f_regression
sns.set()

dataset = pd.read_csv("1.02. Multiple linear regression.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

lr = LinearRegression()
lr.fit(X, y)

r2 = 1 - lr.score(X, y)
n = X.shape[0]
p = X.shape[1]

adjusted_r2 = 1 - r2 * (n-1)/(n-p-1)

p_values = f_regression(X, y)[1].round(3)

lr_summary = pd.DataFrame(data=X.columns.values, columns=["Features"])
lr_summary["Coefficients"] = lr.coef_
lr_summary["p-values"] = p_values.round(3)
print(lr_summary)