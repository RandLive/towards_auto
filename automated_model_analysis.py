# -*- coding: utf-8 -*-
"""
Ver: 00.00.01 Very basic funcitons
Ver: 00.00.02 Change stacking method for fast experiment

@author: ML
"""

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

# In[laoding dataset]
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
y = np.asarray(data.target)
X = np.asarray(data.data)

# always make to this format, maybe larger data
X = X[:500]
y = y[:500]


# In[algorithms]
from sklearn import linear_model
from sklearn import neural_network
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import naive_bayes

import lightgbm as lgb

models = {
        'ridge  ': linear_model.Ridge(alpha=.5,max_iter=1e8),
        'ridgeCV': linear_model.RidgeCV(cv=3),
        'lasso  ': linear_model.Lasso(alpha=1e-6, max_iter=1e8),
        'lr     ': linear_model.LogisticRegression(solver='lbfgs', max_iter=1e4),
        'lrCV   ': linear_model.LogisticRegressionCV(solver='lbfgs', max_iter=1e4, cv=5),
        'mlp_clf': neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(256, 64), random_state=1),
        'svc    ': svm.SVC(),
        'rf     ': ensemble.RandomForestRegressor(max_depth=6),
        'lgbclf ': lgb.LGBMClassifier(gamma='auto', num_leaves=31,learning_rate=0.001, n_estimators=2000),
        'lgbreg ': lgb.LGBMRegressor(gamma='auto', num_leaves=31,learning_rate=0.001, n_estimators=2000),
        'knn    ': neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=15),
        'nb     ': naive_bayes.GaussianNB(),
          }

print('\nall models: ', list(models.keys()))

# In[all scalers]
from sklearn import preprocessing
from sklearn import feature_selection

scalers = {
        'standards':preprocessing.StandardScaler(),
        'minmaxs':preprocessing.MinMaxScaler(),
        'robusts':preprocessing.RobustScaler()
          }

feature_selection = {
        'variance_threshold':feature_selection.VarianceThreshold(threshold=1e-4),        
        }

print('all scalers: ', list(scalers.keys()), '\n')

# In[Preprocessing Pipline]
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=2019, shuffle=True)

pipeline = make_pipeline(
                         scalers['robusts'], 
#                         scalers['standards'],
                         feature_selection['variance_threshold'],
                         )
X = pipeline.fit_transform(X)

mse_score = []
auc_score = []
valid_score = {}
oof=y*0
idx2 = 0
X_s = pd.DataFrame() 

for idx, model in enumerate(models.items()):
    print(idx)
    
    model = model[1]
 
    try:
        print('trainning in: ', list(models.keys())[idx])
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]            
            
            if (list(models.keys())[idx] == 'lgbclf ') | (list(models.keys())[idx] == 'lgbreg '):
                model.fit(X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        eval_metric='l1',
                        early_stopping_rounds=5,
                        verbose=0)               
            else:                  
                model.fit(X_train, y_train)
            
            oof[test_index] = model.predict(X_test)
            
        mse_score.append( mean_squared_error(y, oof) )
        try:
            auc_score.append( roc_auc_score(y, oof) )
        except:
            auc_score.append(0)     
            
        X_s[list(models.keys())[idx]] = oof
            
        valid_score.update({list(models.keys())[idx]: ['mse: ', "{0:.4f}".format(mse_score[idx2]), 'auc: ', "{0:.4f}".format(auc_score[idx2])]})
        
        idx2 += 1
        
    except:
        print('error     in: ', list(models.keys())[idx])
        print('-------------------------------------------')
        
    
# In[showing result]       
print('\nvalid score: \n', pd.DataFrame(valid_score).transpose())

inds = np.argmax(mse_score) 
print('\nmse score - worst model found: ',  list(models.keys())[inds], np.max(mse_score))
inds = np.argmin(auc_score) 
print('auc score - worst model found: ',  list(models.keys())[inds], np.min(auc_score))

inds = np.argmin(mse_score) 
print('mse score - best model found: ',  list(models.keys())[inds], np.min(mse_score))
inds = np.argmax(auc_score) 
print('auc score - best model found: ',  list(models.keys())[inds], np.max(auc_score))

# In[Stacking]

X_s = np.asarray(X_s)
for train_index, test_index in kf.split(X_s):

    X_train, X_test = X_s[train_index], X_s[test_index]
    y_train, y_test = y[train_index], y[test_index]            
    
    try:
        model = models['lrCV   ']
        model.fit(X_train, y_train)
    except:
        model = models['ridgeCV']
        model.fit(X_train, y_train)    
        
    oof[test_index] = model.predict(X_test)

# In[showing result]        
mse_score_s = mean_squared_error(y, oof)
print('\rmse score - stacking: ', mse_score_s) 
try:
    auc_score_s = roc_auc_score(y, oof)
    print('auc score - stacking: ', auc_score_s)
except:
    pass