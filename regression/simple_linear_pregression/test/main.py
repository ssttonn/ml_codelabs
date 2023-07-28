import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


dataset = pd.read_csv("test.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pre = lr.predict(X_test)

plt.scatter(X_train, y_train)
plt.plot(X_train, lr.predict(X_train), color="red")
plt.show()

plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color="red")
plt.show()


