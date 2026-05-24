import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
data = pd.read_csv("machdata.csv")
x = data[["temp"]].values
y = data[["fuel"]].values
model = KNeighborsRegressor(n_neighbors=3)
model.fit(x, y)
print("Predicted value of fuel for 58degC is:", model.predict([[58]]),"L/hr")
print("k value:", model.n_neighbors)