
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time


#data = pd.read_csv('mldata_log_est.csv')
data = pd.read_csv('mldata_lin_est.csv')
x = data.iloc[:,:-1].values
y = data['redshift'].values
y = y.reshape(y.shape[0],1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
sc1 = StandardScaler()
sc1.fit(y_train)
y_train_std = sc1.transform(y_train)
y_test_std = sc1.transform(y_test)

threeD = PolynomialFeatures(degree=3)
x_train_threeD_std = threeD.fit_transform(x_train_std)
x_test_threeD_std = threeD.fit_transform(x_test_std)

start_time = time.time()
non_reg = LinearRegression()
non_reg.fit(x_train_threeD_std, y_train_std)
print(" Running time: %s seconds " % (time.time() - start_time))
y_test_pred_std = non_reg.predict(x_test_threeD_std)
y_train_pred_std = non_reg.predict(x_train_threeD_std)

y_test_pred = sc1.inverse_transform(y_test_pred_std)
y_train_pred = sc1.inverse_transform(y_train_pred_std)

error_train = mean_squared_error(y_train, y_train_pred)
error_test = mean_squared_error(y_test, y_test_pred)
print("Error of training data", error_train)
print("Error of testing data", error_test)

r2_train = r2_score (y_train, y_train_pred)
r2_test = r2_score (y_test, y_test_pred)
print("R2 score of training data", r2_train)
print("R2 score of testing data", r2_test)



