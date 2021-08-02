# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 02:03:35 2021

@author: a0956
"""
# AUC 使用錯誤

# 與準確率不同，必須一次在數據集上計算 AUC，從數學上講，它不等於計算小批量並平均結果。


import numpy as np
from sklearn.metrics import roc_auc_score

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1,0.4, 0.35 , 0.8])
auc0 = roc_auc_score(y_true, y_scores)
print('true auc:',auc0)
y_true = np.array([0, 1])
y_scores = np.array([0.1, 0.8])
auc1=roc_auc_score(y_true, y_scores)
y_true = np.array([0, 1])
y_scores = np.array([0.4, 0.35])
auc2=roc_auc_score(y_true, y_scores) 
print('averaged auc',(auc1+auc2)/2)