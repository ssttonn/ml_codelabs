import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Reading the data from the csv file
dataframe = pd.read_csv("Data.csv")

# Splitting the data into independent and dependent variables
# The independent variables are all the columns except the last column
x = dataframe.iloc[:, :-1].values
# The dependent variable is the last column
y = dataframe.iloc[:, -1].values

# Replacing missing data with the mean of each numerical data column
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# Replace the missing data with the mean of each column
x[:, 1:] = imputer.fit_transform(x[:, 1:])

# Transforming categorical data to numerical data that can be used by the model
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
x = ct.fit_transform(x)

# Transforming the dependent variable, y, to numerical data that can be used by the model
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the data into training and testing data, 80% used for training and 20% used for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

# Feature scaling
sc = StandardScaler()
# Don't include the first 3 columns because it is a categorical data column and have already been transformed
sc.fit()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
print(y_train)
print(y_test)




