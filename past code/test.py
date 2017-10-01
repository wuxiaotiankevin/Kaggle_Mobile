# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 00:18:50 2016

@author: Xiaotian Wu
"""
#%%
import sys

sys.path.append('C:/Users/Xiatian Wu/Anaconda3/Lib/xgboost/xgboost_wrapper.dll')
import xgboost as xgb
iris = load_iris()
DTrain = xgb.DMatrix(iris.data, iris.target)
x_parameters = {"max_depth":[2,4,6]}
xgb.cv(x_parameters, DTrain)
#%%

import xgboost as xgb

#%%

import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']


#%%
import xgboost as xgb
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
DTrain = xgb.DMatrix(iris.data, iris.target)
x_parameters = {"max_depth":6}
xgb.cv(x_parameters, DTrain)