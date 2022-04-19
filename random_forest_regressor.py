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
import matplotlib.pyplot as plt

dfEst = pd.read_csv("mldata_log_est.csv")

colnames = list(dfEst.columns)
X = dfEst[colnames[:-1]]
y = dfEst[colnames[-1]]

# testing appropriate test size

# Going to get MSE and R2 for each fold

n_arr = np.linspace(100,1000,4,dtype=int)
color = ["r","g","b", "m", "k"]
arr_split = np.linspace(0.1,0.5,5)
print("train size:",arr_split)

fig = plt.figure(dpi=120, figsize=(4,6),constrained_layout=False)
spec = fig.add_gridspec(ncols=1, nrows=2, hspace=0.0)

ax0 = fig.add_subplot(spec[0])
ax1 = fig.add_subplot(spec[1])

for ni,n in enumerate(n_arr):
    MSE_list = []
    R2_list = []
    for val_s in arr_split:
        rfr = RandomForestRegressor(n_estimators=n, n_jobs=5,random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=val_s)
        
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_test)
        
        MSE_list.append( mean_squared_error(y_test, y_pred) )
        R2_list.append( r2_score(y_test, y_pred) )
  
    ax0.plot(arr_split, MSE_list, color[ni], label=f"n_est={n}")
    ax1.plot(arr_split, R2_list, color[ni], label=f"n_est={n}")

ax0.set_title("Random Forest Regression")
ax0.set_ylabel("MSE")
ax1.set_ylabel(r"$R^2$")
ax1.set_xlabel("test size")
ax1.legend(loc=0, fontsize=6)
fig.tight_layout()
fig.savefig("figs/rfr_metrics.png", format="png")

#print("MSE:",MSE_list)
#print("R2:",R2_list)
    
    
    
