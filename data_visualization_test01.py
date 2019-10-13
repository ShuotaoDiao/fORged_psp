import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
hardware_device = pd.read_csv("/Users/sonny/Documents/INFORMS_Project/Ten-Year-Demand.csv")
print(hardware_device .head())
# rename the column
hardware_device.columns = ['Year','Month','Demand']
print(hardware_device.head())
num_dataPoint = hardware_device.shape[0]
# tackle empty entries in the column of Year
# not necessary
'''
print(hardware_device.shape)
num_dataPoint = hardware_device.shape[0]
print(hardware_device.Demand[0])
year_ifNaN = hardware_device.Year.isna()
print(year_ifNaN)
for index_dataPoint in range(num_dataPoint):
    if year_ifNaN[index_dataPoint]:
        hardware_device.Year[index_dataPoint] = hardware_device.Year[index_dataPoint - 1]
print(hardware_device.head())
'''
# original series
Z = hardware_device.Demand
plt.plot(Z)
#plt.show()

# detrend
T = np.array([_ for _ in range(num_dataPoint)]).reshape((num_dataPoint,1))
# create a linear model
linear_model = LinearRegression()
linear_model.fit(T,Z)
Z_trend = linear_model.predict(T)
Z_DT = Z - Z_trend
plt.plot(Z_trend)
plt.show()
print(linear_model.predict(np.array(num_dataPoint).reshape((1,-1))))
plt.plot(Z_DT)
plt.show()

# AR model
AR_model = ARIMA(Z_DT,order=(2,0,0))
AR_model_fit = AR_model.fit()
print(AR_model_fit.summary())
plt.plot(Z_DT)
plt.plot(AR_model_fit.fittedvalues, color='red')
plt.show()
print(Z_DT[-2:])
print(AR_model_fit.forecast()[0])
print(AR_model_fit.params)
print(AR_model_fit.bse)
print(AR_model_fit.params[0])
