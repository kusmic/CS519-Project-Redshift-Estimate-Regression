#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:12:19 2022

@author: Samir
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

val=2.1
sns.set(font_scale=val)

def pairplot_lin():
    df_lin = pd.read_csv("mldata_lin_est.csv")
    colnames_lin = np.asarray(list(df_lin.columns))
    print(colnames_lin)
    sns.pairplot(df_lin[colnames_lin], height=3, aspect=1., kind="hist")
    plt.tight_layout()
    plt.savefig("figs/pairplot_log.png", format="png")
    plt.show()

def pairplot_log():
    df_log = pd.read_csv("mldata_log_est.csv")
    colnames_log = np.asarray(list(df_log.columns))
    #print(colnames_log)
    print(colnames_log)
    sns.pairplot(df_log[colnames_log], height=3, aspect=1., kind="hist")
    plt.tight_layout()
    plt.savefig("figs/pairplot_log.png", format="png")
    plt.show()
    
pairplot_log()
