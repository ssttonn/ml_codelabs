import pandas as pd


dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

