# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:02:26 2019

@author: meli
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV


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


def grid_cv(X, y, clf, parameters, nfolds=5):
    clf2 = GridSearchCV(clf, parameters, cv=nfolds)
    clf2.fit(X, y)
    return clf2.best_params_ 


#from sklearn import svm
#clf = svm.SVC()
#parameters = {'kernel':['linear', 'poly', 'rbf'], 'C':[1, 10, 100]}
#clf_param = grid_cv(X, y, clf, parameters, 5)
#print('SVC para: ', clf_param)


from sklearn import linear_model
clf = linear_model.Ridge()
parameters = {'alpha':[0.1, 1, 10], 'normalize':[True, False]}
clf_param = grid_cv(X, y, clf, parameters, 5)
print('Ridge para: ', clf_param)

clf = linear_model.LogisticRegression()
parameters =  {'solver':['newton-cg', 'lbfgs', 'liblinear'],'warm_start':[True, False]}
clf_param = grid_cv(X, y, clf, parameters, 5)
print('lr para: ', clf_param)


from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
parameters =  {'n_neighbors':[2, 5, 10, 30, 50, 100]}
clf_param = grid_cv(X, y, clf, parameters, 5)
print('neighbors para: ', clf_param)

from sklearn import ensemble
clf = ensemble.RandomForestRegressor()
parameters =  {'max_depth':[3, 7, 10, 30]}
clf_param = grid_cv(X, y, clf, parameters, 5)
print('rfreg para: ', clf_param)


clf = ensemble.RandomForestClassifier()
parameters =  {'max_depth':[3, 7, 10, 30]}
clf_param = grid_cv(X, y, clf, parameters, 5)
print('rfclf para: ', clf_param)


from sklearn import neural_network
clf = neural_network.MLPClassifier(max_iter=10000)
parameters =  {'hidden_layer_sizes':[[256, 64], [256, 64, 32, 32, 32], [16,16,2]]}
clf_param = grid_cv(X, y, clf, parameters, 5)
print('nnclf para: ', clf_param)

clf = neural_network.MLPRegressor(max_iter=10000)
parameters =  {'hidden_layer_sizes':[[256, 64], [256, 64, 32, 32, 32], [16,16,2]]}
clf_param = grid_cv(X, y, clf, parameters, 5)
print('nnreg para: ', clf_param)