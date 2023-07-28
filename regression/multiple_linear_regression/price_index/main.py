import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
from joblib import dump, load

data = {'year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
        'month': [12,11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
        'interest_rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
        'unemployment_rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
        'index_price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]
        }

dataframe = pd.DataFrame(data)

X = dataframe[["interest_rate", "unemployment_rate"]]
y = dataframe["index_price"]

lr = LinearRegression()
lr.fit(X, y)

print('Intercept: \n', lr.intercept_)
print('Coefficients: \n', lr.coef_)

X = add_constant(X)

ols_model = OLS(y, X).fit()

predictions = ols_model.predict(X)

print(ols_model.summary())