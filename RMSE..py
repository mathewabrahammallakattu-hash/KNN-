import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import math
real_data = pd.read_csv("machdata.csv")
x= real_data[['temp' ]]
y= real_data[['fuel']]
model = LinearRegression()
model.fit(x,y)
print(model.coef_)
print(model.intercept_)

ypred = model.predict(x)
mse = mean_squared_error(y, ypred)  
rmse = math.sqrt(mse)
print("mse:", mse)
print("Root Mean Squared Error:", rmse)
