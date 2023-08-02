import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()

dataset = pd.read_csv("1.02. Multiple linear regression.csv")
print(dataset)

y = dataset["GPA"]
X1 = dataset[["SAT", "Rand 1,2,3"]]

X = sm.add_constant(X1)
results = sm.OLS(y, X).fit()
print(results.summary())