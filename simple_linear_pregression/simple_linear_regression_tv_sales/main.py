import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("tvmarketing.csv")

x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

plt.scatter(x, y, color="#1f77b4")
plt.title("TV Marketing data visualization")
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()
