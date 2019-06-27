# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:02:26 2019

@author: meli
"""

import pandas as pd
import numpy as np

from models_ml import auto_model


# In[laoding dataset]
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
y = np.asarray(data.target)
X = np.asarray(data.data)
X_pred = np.asarray(data.data)
sub = pd.DataFrame(X)
sub['target'] = y

# always make to this format, maybe larger data
#tmp = pd.read_csv('../input/train.csv')
#y = tmp.target.values
#tmp.drop(['id', 'target'], axis=1, inplace=True)
#X = np.asarray(tmp)
#
#sub = pd.read_csv('../input/sample_submission.csv')
#tmp = pd.read_csv('../input/test.csv')
#tmp.drop('id', axis=1, inplace=True)
#X_pred = np.asarray(tmp)


# In[Auto ML, all data in array form]
nfolds = 2
auto_model(X, y, X_pred, sub, nfolds)