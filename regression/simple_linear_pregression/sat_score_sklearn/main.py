import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
sns.set()

dataset = pd.read_csv("1.01. Simple linear regression.csv")
X = dataset[["SAT"]].values
y = dataset["GPA"].values

lr = LinearRegression()
lr.fit(X, y)
y_pre = lr.predict(X)

new_data = pd.DataFrame(data=[1740, 1760], columns=["SAT"])

new_data["Predicted GPA"] = lr.predict(new_data)

print(new_data)