import pandas as pd
import statsmodels.api as sm
import numpy as np

dataset = pd.read_csv("Example_bank_data.csv")
dataset["y"] = dataset["y"].map({"yes": 1, "no": 0})
X = dataset["duration"]
y = dataset["y"]

X = sm.add_constant(X)
reg_log = sm.Logit(y, X)
results_log = reg_log.fit()
print(results_log.summary())

print(np.exp(-1.7001 + 0.0051)/ (1 + np.exp(-1.7001 + 0.0051)))