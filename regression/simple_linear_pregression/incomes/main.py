import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataframe = pd.read_csv("income_data.csv")

x = dataframe.iloc[:, 1:-1].values
y = dataframe.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pre = lr.predict(x_test)


