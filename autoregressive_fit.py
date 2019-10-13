import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

hardware_device = pd.read_csv("/Users/sonny/Documents/INFORMS_Project/Ten-Year-Demand.csv")
# rename the column
hardware_device.columns = ['Year','Month','Demand']
print(hardware_device.head())
num_dataPoint = hardware_device.shape[0]
# detect the trend of the data
T = np.array([_ for _ in range(num_dataPoint)]).reshape((num_dataPoint,1))
Z = hardware_device.Demand
plt.plot(T, Z, color = 'blue')
plt.xlabel("Time")
plt.ylabel("Demand (Linear Trend Removed)")

plt.show()

# polynomial fitting
print("Number of data points: {}".format(num_dataPoint))

# create a linear model
linear_model = LinearRegression()
linear_model.fit(T,Z)
Z_trend = linear_model.predict(T)
Z_DT = Z - Z_trend
plt.plot(Z_trend)
plt.show()
print(linear_model.predict(np.array(num_dataPoint).reshape((1,-1))))
plt.plot(Z_DT)
plt.xlabel("Time")
plt.ylabel("Demand (Linear Trend Removed)")
plt.show()

# AR model
AR_model1 = ARIMA(Z_DT,order=(1,0,0))
AR_model1_fit = AR_model1.fit()
print(AR_model1_fit.summary())
plt.plot(Z_DT)
plt.plot(AR_model1_fit.fittedvalues, color='m')
plt.xlabel("Time")
plt.ylabel("Demand (Linear Trend Removed)")
plt.title("p = 1")
plt.show()

AR_model2 = ARIMA(Z_DT,order=(2,0,0))
AR_model2_fit = AR_model2.fit()
print(AR_model2_fit.summary())
plt.plot(Z_DT)
plt.plot(AR_model1_fit.fittedvalues, color='m')
plt.xlabel("Time")
plt.ylabel("Demand (Linear Trend Removed)")
plt.title("p = 2")
plt.show()

AR_model3 = ARIMA(Z_DT,order=(3,0,0))
AR_model3_fit = AR_model3.fit()
print(AR_model3_fit.summary())
plt.plot(Z_DT)
plt.plot(AR_model1_fit.fittedvalues, color='m')
plt.xlabel("Time")
plt.ylabel("Demand (Linear Trend Removed)")
plt.title("p = 3")
plt.show()