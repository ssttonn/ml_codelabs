import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X, y)

plt.scatter(X, y, color="blue")
plt.plot(X, dtr.predict(X), color="r")
plt.show()
