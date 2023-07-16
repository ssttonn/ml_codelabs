import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1, 1)

ss_x = StandardScaler()
X_trans = ss_x.fit_transform(X)
ss_y = StandardScaler()
y_trans = np.ravel(ss_y.fit_transform(y))

svr = SVR(kernel="rbf")
svr.fit(X_trans, y_trans)

y_pre = ss_y.inverse_transform(svr.predict(ss_x.transform(X)).reshape(-1, 1))

plt.scatter(X, y)
plt.plot(ss_x.inverse_transform(X_trans), y_pre, color="red")
plt.show()

