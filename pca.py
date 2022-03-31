#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 21:26:37 2022

@author: Samir
"""

from sklearn.decomposition import PCA, KernelPCA
import pandas as pd

df = pd.read_csv("mldata_log_est.csv")

colnames = list(df.columns)
#print(df[colnames[:-1]])
X = df[colnames[:-1]].to_numpy()

pca = PCA(n_components=4) # 4 PCs -> 0.9999999999999944 total explained variance ratio
pca.fit(X)
print(pca.explained_variance_ratio_.sum())

#kpca = KernelPCA(n_components=4,kernel="rbf")
#kpca.fit(X)
#print(kpca.eigenvalues_)