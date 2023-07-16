import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("Data.csv")

x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

si = SimpleImputer(missing_values=np.nan, strategy="mean")
x[:, 1:] = si.fit_transform(x[:, 1:])

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
x = ct.fit_transform(x)

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,random_state=1)

sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
print(y_train)
print(y_test)