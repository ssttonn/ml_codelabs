import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

dataset = pd.read_csv("1.03. Dummies.csv")
print(dataset)

data = dataset.copy()
data["Attendance"] = data["Attendance"].map({"Yes": 1, "No": 0})

print(data)
print(data.describe())

y = data["GPA"]
x1 = data[["SAT", "Attendance"]]

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())

plt.scatter(data["SAT"], data["GPA"], c=data["Attendance"], cmap="RdYlGn_r")
yhat_no = 0.6439 + 0.0014*data["SAT"]
yhat_yes = 0.8665 + 0.0014*data["SAT"]
yhat = 0.0017*data["SAT"] + 0.275
fig = plt.plot(data["SAT"], yhat_no, lw=2, c="#006837", label="Regression line 1")
fig = plt.plot(data["SAT"], yhat_yes, lw=2, c="#a50026", label="Regression line 2")
fig = plt.plot(data["SAT"], yhat, lw=2, c="#4C72B0", label="Regression line")
plt.xlabel("SAT", fontsize = 20)
plt.ylabel("GPA", fontsize = 20)
plt.show()

new_data = pd.DataFrame({'const': 1, "SAT": [1700, 1670], "Attendance": [0, 1]})
new_data = new_data[["const", "SAT", "Attendance"]]
predictions = results.predict(new_data)
predictions_data = pd.DataFrame(predictions)
joined_data = new_data.join(predictions_data)

print(joined_data)