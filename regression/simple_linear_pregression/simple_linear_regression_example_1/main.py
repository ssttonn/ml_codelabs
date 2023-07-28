import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from threading import Thread

dataframe = pd.read_csv("Salary_Data.csv")

x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

lr = LinearRegression()
lr.fit(x, y)
y_pre = lr.predict(x_test)


def plot_train_data(axes):
    axes.scatter(x_train, y_train, color="red")
    axes.plot(x_train, lr.predict(x_train), color="blue")
    axes.set_title("Salary vs Experience (Training set)")
    axes.set_xlabel("Years of Experience")
    axes.set_ylabel("Salary")


def plot_test_data(axes):
    axes.scatter(x_test, y_test, color="red")
    axes.plot(x_train, lr.predict(x_train), color="blue")
    axes.set_title("Salary vs Experience (Test set)")
    axes.set_xlabel("Years of Experience")
    axes.set_ylabel("Salary")


figure, axis = plt.subplots(1, 2)
plot_train_data(axis[0])
plot_test_data(axis[1])
plt.show()

