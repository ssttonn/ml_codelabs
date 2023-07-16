import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("insurance.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, -2] = le.fit_transform(X[:, -2])

ct = ColumnTransformer([("encoder", OneHotEncoder(), [-1])], remainder="passthrough")
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pre = lr.predict(X_test)

print(y_pre)
print(y_test)
