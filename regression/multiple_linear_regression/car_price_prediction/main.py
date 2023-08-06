import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
sns.set()
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

dataset = pd.read_csv("car_price.csv")

data = dataset.drop(["Model"], axis=1)

# Drop all rows contain na or null features
data_no_mv = data.dropna(axis=0)

# Take the price at 99% quantile in Price Distribution
q = data_no_mv["Price"].quantile(0.99)
data_no_mv = data_no_mv[data_no_mv["Price"] < q]

q = data_no_mv["Mileage"].quantile(0.99)
data_no_mv = data_no_mv[data_no_mv["Mileage"]<q]

# 6.5 is the limit for normal engine value, should lower than 6.5
q = 6.5
data_no_mv = data_no_mv[data_no_mv["EngineV"] < q]

# Need to reset the index because the dataset still keeps old indexes when drop rows
data_no_mv = data_no_mv.reset_index(drop=True)
data_no_mv["Log Price"] = np.log(data_no_mv["Price"])
data_no_mv.drop(["Price"], axis=1)

variables = data_no_mv[["Mileage", "Year", "EngineV"]]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns
print(vif)

data_no_mv = data_no_mv.drop(["Year", "Price"], axis=1)
data_no_mv = pd.get_dummies(data_no_mv, drop_first=True)

X = data_no_mv.drop(["Log Price"], axis=1)
y = data_no_mv["Log Price"]

sc = StandardScaler()
X[['Mileage', 'EngineV']] = sc.fit_transform(X[['Mileage', 'EngineV']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_hat = lr.predict(X_train)

y_hat_test = lr.predict(X_test)

reg_summary = pd.DataFrame(X.columns.values, columns=["Features"])
reg_summary["Weight"] = lr.coef_

df_pf = pd.DataFrame(np.exp(y_hat_test).reshape(-1), columns=["Predictions"])
df_pf["Target"] = np.exp(y_test.reset_index(drop=True))
df_pf["Residual"] = df_pf["Target"] - df_pf["Predictions"]
df_pf["Difference%"] = np.absolute(df_pf["Residual"] / df_pf["Target"] * 100)
df_pf.sort_values(by=["Difference%"], inplace=True)
# print(df_pf)
# plt.scatter(y_test, y_hat_test, alpha=0.2)
# plt.xlabel("Targets (y_test)", size=18)
# plt.ylabel("Predictions (y_hat_test)", size=18)
# plt.xlim(6, 13)
# plt.ylim(6, 13)
# plt.show()
#
# sns.distplot(y_test-y_hat_test)
# plt.xlabel("Error", size=18)
# plt.show()

