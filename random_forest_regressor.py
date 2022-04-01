#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:22:15 2022

@author: Samir
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

dfEst = pd.read_csv("mldata_log_est.csv")

colnames = list(dfEst.columns)
X = dfEst[colnames[:-1]]
y = dfEst[colnames[-1]]

# testing appropriate test size

# Going to get MSE and R2 for each fold

arr_split = np.linspace(0.1,0.5,5)
print("train size:",arr_split)
MSE_list = []
R2_list = []

for val_s in arr_split:
    rfr = RandomForestRegressor(n_estimators=1000)

    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=val_s)
    
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    
    MSE_list.append( mean_squared_error(y_test, y_pred) )
    R2_list.append( r2_score(y_test, y_pred) )
    
print("MSE:",MSE_list)
print("R2:",R2_list)
    
    
    
