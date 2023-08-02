import statsmodels.api as sm
import pandas as pd

dataset = pd.read_csv("Startups.csv")
X1 = dataset[["Marketing Expenditure"]]
X2 = dataset[["Marketing Expenditure", "R&D Expenditure", "Administration Expenditure"]]
y = dataset["Profit"]

X1 = sm.add_constant(X1)
results1 = sm.OLS(y, X1).fit()
print(results1.summary())

X2 = sm.add_constant(X2)
results2 = sm.OLS(y, X2).fit()
print(results2.summary())

yhat1 = 6e+04 + 0.2465*X1

yhat2 = 5.012e+04 + 0.0272*420000 + 0.8057*125000 - 0.0268*120000

print(yhat2)