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
plt.ylabel("Demand")
plt.show()

# polynomial fitting
print("Number of data points: {}".format(num_dataPoint))

# split the dataset into training set and test test
Z_train = Z[0:96]
Z_test = Z[96:]
T_train = T[0:96]
T_test = T[96:]

# order is 2
order2 = 2
poly_feature_reg2 = PolynomialFeatures(degree=order2)
T_train_poly2 = poly_feature_reg2.fit_transform(T_train)
T_test_poly2 = poly_feature_reg2.fit_transform(T_test)
T_poly2 = poly_feature_reg2.fit_transform(T)
poly_reg2 = LinearRegression()
poly_reg2.fit(T_train_poly2, Z_train)
Z_train_pred_reg2 = poly_reg2.predict(T_train_poly2)
Z_test_pred_reg2 = poly_reg2.predict(T_test_poly2)
plt.plot(T_train,Z_train,color='b')
plt.plot(T_train,Z_train_pred_reg2,color = 'r')
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("Order of the polynomial regression is 2")
plt.show()
# coefficients
print("Order of the polynomial regression is 2")
print("Coefficients: \n", poly_reg2.coef_)
print("Mean squared error: {}".format(mean_squared_error(Z_test,Z_test_pred_reg2)))
print("R2 score: {}".format(r2_score(Z_test,Z_test_pred_reg2)))
# order is 3
order3 = 3
poly_feature_reg3 = PolynomialFeatures(degree=order3)
T_train_poly3 = poly_feature_reg3.fit_transform(T_train)
T_test_poly3 = poly_feature_reg3.fit_transform(T_test)
T_poly3= poly_feature_reg3.fit_transform(T)
poly_reg3 = LinearRegression()
poly_reg3.fit(T_train_poly3, Z_train)
Z_train_pred_reg3 = poly_reg3.predict(T_train_poly3)
Z_test_pred_reg3 = poly_reg3.predict(T_test_poly3)
plt.plot(T_train,Z_train,color='b')
plt.plot(T_train,Z_train_pred_reg3,color = 'r')
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("Order of the polynomial regression is 3")
plt.show()
# coefficients
print("Order of the polynomial regression is 3")
print("Coefficients: \n", poly_reg3.coef_)
print("Performance in the test set")
print("Mean squared error: {}".format(mean_squared_error(Z_test,Z_test_pred_reg3)))
print("R2 score: {}".format(r2_score(Z_test,Z_test_pred_reg3)))

# order is 4
order4 = 4
poly_feature_reg4 = PolynomialFeatures(degree=order4)
T_train_poly4 = poly_feature_reg4.fit_transform(T_train)
T_test_poly4 = poly_feature_reg4.fit_transform(T_test)
T_poly4= poly_feature_reg4.fit_transform(T)
poly_reg4 = LinearRegression()
poly_reg4.fit(T_train_poly4, Z_train)
Z_train_pred_reg4 = poly_reg4.predict(T_train_poly4)
Z_test_pred_reg4 = poly_reg4.predict(T_test_poly4)
plt.plot(T_train,Z_train,color='b')
plt.plot(T_train,Z_train_pred_reg4,color = 'r')
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("Order of the polynomial regression is 4")
plt.show()
# coefficients
print("Order of the polynomial regression is 4")
print("Coefficients: \n", poly_reg4.coef_)
print("Performance in the test set")
print("Mean squared error: {}".format(mean_squared_error(Z_test,Z_test_pred_reg4)))
print("R2 score: {}".format(r2_score(Z_test,Z_test_pred_reg4)))

# order is 5
order5 = 5
poly_feature_reg5 = PolynomialFeatures(degree=order5)
T_train_poly5 = poly_feature_reg5.fit_transform(T_train)
T_test_poly5 = poly_feature_reg5.fit_transform(T_test)
T_poly5 = poly_feature_reg5.fit_transform(T)
poly_reg5 = LinearRegression()
poly_reg5.fit(T_train_poly5, Z_train)
Z_train_pred_reg5 = poly_reg5.predict(T_train_poly5)
Z_test_pred_reg5 = poly_reg5.predict(T_test_poly5)
plt.plot(T_train,Z_train,color='b')
plt.plot(T_train,Z_train_pred_reg5,color = 'r')
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("Order of the polynomial regression is 5")
plt.show()
# coefficients
print("Order of the polynomial regression is 5")
print("Coefficients: \n", poly_reg5.coef_)
print("Performance in the test set")
print("Mean squared error: {}".format(mean_squared_error(Z_test,Z_test_pred_reg5)))
print("R2 score: {}".format(r2_score(Z_test,Z_test_pred_reg5)))
print(Z_test)
print(Z_test_pred_reg5)


# linear fitting
linear_model = LinearRegression()
linear_model.fit(T_train,Z_train)
Z_train_pred = linear_model.predict(T_train)
Z_test_pred = linear_model.predict(T_test)
plt.plot(T_train,Z_train,color='b')
plt.plot(T_train,Z_train_pred,color = 'r')
plt.xlabel("Time")
plt.ylabel("Demand")
plt.title("Linear Regression")
plt.show()
# coefficients
print("Linear Regression")
print("Coefficients: \n", linear_model.coef_)
print("Performance in the test set")
print("Mean squared error: {}".format(mean_squared_error(Z_test,Z_test_pred)))
print("R2 score: {}".format(r2_score(Z_test,Z_test_pred)))
print(Z_test)
print(Z_test_pred)