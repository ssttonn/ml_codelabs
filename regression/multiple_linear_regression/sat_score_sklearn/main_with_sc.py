import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("1.02. Multiple linear regression.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

sc = StandardScaler()
X = sc.fit_transform(X)

lr = LinearRegression()
lr.fit(X, y)

lr_summary = pd.DataFrame([["Intercept", lr.intercept_], ["SAT", lr.coef_[0]], ["Rand 1,2,3", lr.coef_[1]]], columns=["Features", "Weights"])

new_data = pd.DataFrame([[1700, 2], [1800, 1]], columns=["SAT", "Rand 1,2,3"])
new_data_scaled = sc.transform(new_data)

y_pred = lr.predict(new_data_scaled)
print(y_pred)