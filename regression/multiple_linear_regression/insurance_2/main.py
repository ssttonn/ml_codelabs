import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
sns.set()
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

dataset = pd.read_csv("Insurance.csv")

dataset["Log Insurance Charges"] = np.log(dataset["Insurance Charges"])

dataset = dataset[['Age', 'BMI', 'Smoker', 'Region', 'Insurance Charges', 'Log Insurance Charges']]
dataset = pd.get_dummies(dataset, drop_first=True, dtype=int).drop([ "Insurance Charges"], axis=1)

X = dataset.drop(["Log Insurance Charges"], axis=1)
y = dataset["Log Insurance Charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train[['Age', 'BMI']] = sc.fit_transform(X_train[['Age', 'BMI']])
X_test[['Age', 'BMI']] = sc.transform(X_test[['Age', 'BMI']])

lr = LinearRegression()
lr.fit(X_train, y_train)
y_hat = lr.predict(X_train)

# plt.scatter(y_train, y_hat)
# plt.xlabel("Targets (y_train)", size=18)
# plt.ylabel("Predictions (y_hat)", size=18)
# plt.xlim(6, 13)
# plt.ylim(6, 13)
# plt.show()
#
# residual = y_hat - y_train
# sns.distplot(residual)
# plt.title("Residual PDF", size=18)
# plt.show()

reg_summary = pd.DataFrame(X.columns.values, columns=["Features"])
reg_summary["Weight"] = lr.coef_
print(reg_summary)

y_hat_test = lr.predict(X_test)

# plt.scatter(y_test, y_hat_test, alpha=0.2)
# plt.xlabel("Targets (y_test)", size=18)
# plt.ylabel("Predictions (y_hat_test)", size=18)
# plt.xlim(6, 13)
# plt.ylim(6, 13)
# plt.show()

y_test = y_test.reset_index(drop=True)
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=["Prediction"])
df_pf["Target"] = np.exp(y_test)
df_pf["Residual"] = df_pf["Target"] - df_pf["Prediction"]
df_pf["Squared Residual"] = df_pf["Residual"]**2
df_pf["Difference%"] = np.absolute(df_pf["Residual"]/df_pf["Target"]*100)
df_pf.sort_values(by=["Difference%"], ascending=True, inplace=True)
print(df_pf)
print(df_pf.describe())


# f, (fig1, fig2) = plt.subplots(1, 2, sharey=True ,figsize=(15, 5))
# fig1.scatter(dataset["Age"], dataset["Log Insurance Charges"])
# fig1.set_title("Age vs Log Insurance Charges")
# fig2.scatter(dataset["BMI"], dataset["Log Insurance Charges"])
# fig2.set_title("BMI vs Log Insurance Charges")
# plt.show()