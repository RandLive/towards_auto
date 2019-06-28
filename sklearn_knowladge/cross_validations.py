# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:15:19 2019
@author: meli
"""

# In[laoding dataset]
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

data = load_breast_cancer()
y = np.asarray(data.target)
X = np.asarray(data.data)
X_pred = np.asarray(data.data)
sub = pd.DataFrame(X)
sub['target'] = y

# In[Kfold]
nfolds = 5

# In[Kfold]
from sklearn.model_selection import KFold 
kf = KFold(n_splits=nfolds, random_state=2019, shuffle=True)
for idx, (train_index, test_index) in enumerate( kf.split(X) ):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]  

# In[Repeated Kfold]    
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=nfolds, n_repeats=5, random_state=2019)
for idx, (train_index, test_index) in enumerate( rkf.split(X) ):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
# In[Leave One Out -> for extrem small dataset]
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for idx, (train_index, test_index) in enumerate( loo.split(X) ):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
# In[Leave P Out]
#from sklearn.model_selection import LeavePOut
#lpo  = LeavePOut(p=2)
#for idx, (train_index, test_index) in enumerate( lpo.split(X) ):
#  print(idx)
#  X_train, X_test = X[train_index], X[test_index]
#  y_train, y_test = y[train_index], y[test_index]

# In[Stratified k-fold]
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=nfolds, random_state=2019, shuffle=True)
for idx, (train_index, test_index) in enumerate( skf.split(X, y) ):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
# In[Shuffled Stratified k-fold]
from sklearn.model_selection import StratifiedShuffleSplit
sskf = StratifiedShuffleSplit(n_splits=nfolds, random_state=2019)
for idx, (train_index, test_index) in enumerate( sskf.split(X, y) ):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
# In[Repeated Stratified KFold]
from sklearn.model_selection import RepeatedStratifiedKFold 
rskf = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=5, random_state=2019)
for idx, (train_index, test_index) in enumerate( rskf.split(X, y) ):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
# In[Group k-fold] etc, plz view: https://scikit-learn.org/stable/modules/cross_validation.html
  
# In[]
from sklearn.model_selection import TimeSeriesSplit 
tkf = TimeSeriesSplit(n_splits=nfolds)
for idx, (train_index, test_index) in enumerate( tkf.split(X, y) ):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]