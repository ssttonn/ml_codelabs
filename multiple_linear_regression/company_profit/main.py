import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

dataframe = pd.read_csv("50_Startups.csv")
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [-1])], remainder="passthrough")
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pre = lr.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pre.reshape(len(y_pre), 1), y_test.reshape(len(y_test), 1)), 1))

