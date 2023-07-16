import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("tvmarketing.csv")

x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0000)


print(np.newaxis)

lr = LinearRegression()
lr.fit(x_train, y_train)

# y = intercept + coef * x
# y = 6.9897 + 0.0464 * x
print(lr.intercept_)
print(lr.coef_)

y_pre = lr.predict(x_test)


plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, lr.predict(x_train), color="blue")
plt.title("TV sales (Training set)")
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()

plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, lr.predict(x_train), color="blue")
plt.title("TV sales (Test set)")
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()

