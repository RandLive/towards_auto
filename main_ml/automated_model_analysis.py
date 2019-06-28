# -*- coding: utf-8 -*-
"""
Ver: 00.00.01 Very basic funcitons
Ver: 00.00.02 Change stacking method for fast experiment
Ver: 00.00.03 add read_csv
Ver: 00.00.04 using NN for stacking instead of sklearn
Ver: 00.00.05 add catboost
Ver: 00.00.06 add keras nn
Ver: 00.00.07 add PowerTransformer preprocessing - make the features more gaussian like

Ver: 00.01.01 Unsupervised pre-trainning
Ver: 01.00.01 Modulation


-----------------------------------
TODO: 0. Add xgboost sklearn 
TODO: 1. selectable metric as metric 3 (custom)
TODO: 2. using Keras NN to stack the models
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

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.01,
                               patience=5,
                               verbose=0,
                               min_delta=1e-5,
                               mode='min')

# In[algorithms]
from sklearn import linear_model
from sklearn import neural_network
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import naive_bayes
#from sklearn import mixture
from sklearn import tree
from catboost import CatBoostRegressor, CatBoostClassifier

import lightgbm as lgb

def auto_model(X, y, X_pred, sub):
    
    models = {
            'ridge  ': linear_model.Ridge(alpha=.5,max_iter=1e8),
            'ridgeCV': linear_model.RidgeCV(cv=3),
            'lasso  ': linear_model.Lasso(alpha=1e-6, max_iter=1e8),
            'lr     ': linear_model.LogisticRegression(solver='lbfgs', max_iter=1e4),
            'lrCV   ': linear_model.LogisticRegressionCV(solver='lbfgs', max_iter=1e4, cv=5),
            'mlp_clf': neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(256, 64, 32, 32, 32), random_state=1),
            'mlp_reg': neural_network.MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(256, 64, 32, 32, 32), random_state=1),
            'svc    ': svm.SVC(),
            'rfreg  ': ensemble.RandomForestRegressor(max_depth=4),
            'rfclf  ': ensemble.RandomForestClassifier(max_depth=4),
            'lgbclf ': lgb.LGBMClassifier(gamma='auto', num_leaves=4,learning_rate=0.001, n_estimators=2000),
            'lgbreg ': lgb.LGBMRegressor(gamma='auto', num_leaves=31,learning_rate=0.001, n_estimators=20000),
            'knn    ': neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=15),
            'nb     ': naive_bayes.GaussianNB(),
            'dt     ': tree.DecisionTreeClassifier(),
    #        'catreg ': CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=3, verbose = True),
    #        'catclf ': CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=2, verbose = True),
            }
    
    print('\nall models: ', list(models.keys()))
    
    # In[all scalers]
    from sklearn import preprocessing
    from sklearn import feature_selection
    from sklearn.decomposition import PCA
    
    preprocessings = {
            'standards':preprocessing.StandardScaler(),
            'minmaxs':preprocessing.MinMaxScaler(),
            'robusts':preprocessing.RobustScaler(),
            'PCA': PCA(),
            'PowerTransformer':preprocessing.Normalizer(), 
            'variance_threshold':feature_selection.VarianceThreshold(threshold=0.5),        
                  }
                 
    print('all preprocessings: ', list(preprocessings.keys()), '\n')
    
    # In[Preprocessing Pipline]
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold
    nfolds = 5
    
    kf = KFold(n_splits=nfolds, random_state=2019, shuffle=True)
    
    pipe_preprocessing = make_pipeline(
    #                        preprocessings['variance_threshold'],
                             preprocessings['standards'],
    #                         preprocessings['minmaxs'],
    #                         preprocessings['robusts'],
    #                         preprocessings['PCA'],
    #                         preprocessings['PowerTransformer'],
                             
                            )
    
    full = np.concatenate((X, X_pred), axis=0)
    pipe_preprocessing.fit(full)
    X = pipe_preprocessing.transform(X)
    X_pred = pipe_preprocessing.transform(X_pred)
    #
    mse_score = []
    auc_score = []
    valid_score = {}
    oof=y*0
    idx2 = 0
    X_s = pd.DataFrame() 
    X_s_preds = pd.DataFrame()
    
    for idx, model in enumerate(models.items()):
        
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
                elif (list(models.keys())[idx] == 'catreg ') | (list(models.keys())[idx] == 'catclf '):
                    model.fit(X_train, y_train, eval_set = (X_test, y_test))                  
                else:                  
                    model.fit(X_train, y_train)
                
                oof[test_index] = model.predict(X_test)
                
            mse_score.append( mean_squared_error(y, oof) )
            try:
                auc_score.append( roc_auc_score(y, oof) )
            except:
                auc_score.append(0)     
                
            X_s[list(models.keys())[idx]] = oof
            X_s_preds[list(models.keys())[idx]] = model.predict(X_pred)
                
            valid_score.update({list(models.keys())[idx]: ['mse: ', "{0:.4f}".format(mse_score[idx2]), 'auc: ', "{0:.4f}".format(auc_score[idx2])]})
            
            idx2 += 1
            
        except:
            print('error     in: ', list(models.keys())[idx])
            print('-------------------------------------------')
            
    # In[Keras NN]
    X_s0 = np.asarray(X)
    X_s0_preds = np.asarray(X_pred)
    
    oof = np.zeros(len(X_s0))
    predictions = np.asarray(sub.target * 0.)
    
    pipe_preprocessing = make_pipeline(
                             preprocessings['minmaxs'],
                             preprocessings['PowerTransformer'],
                             )
    
    full = np.concatenate((X_s0, X_s0_preds), axis=0)
    pipe_preprocessing.fit(full)
    X = pipe_preprocessing.transform(X_s0)
    X_pred = pipe_preprocessing.transform(X_s0_preds)
    #
    print('\nstart kerasNN ... ')
    
    for fold_, (train_index, test_index) in enumerate(kf.split(X)):
        
        model = Sequential([
                Dense(256, input_shape=(X_s0.shape[1],)),
                Activation('relu'),
                Dense(128),
                Activation('relu'),
                Dense(64),
                Activation('relu'),
                Dense(32),
                Activation('relu'),
                Dense(32),
                Activation('relu'),
                Dense(1),
                Activation('sigmoid'),
                ])
        
        model.compile(optimizer='adam',
                          loss='binary_crossentropy')
        
        file_path="NN_ml_"+"_model_"+"loop_"+str(fold_)+".hdf5"
        
        X_tr, X_val = X_s0[train_index], X_s0[test_index]
        y_tr, y_val = y[train_index], y[test_index]
        
        callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=20),
                     ModelCheckpoint(filepath=file_path, monitor='val_loss', mode='min', save_best_only=True),
                     lr_reduced]
        
        model.fit(X_tr, y_tr, 
                       epochs=750,
                       batch_size=512, 
                       callbacks=callbacks,
                       shuffle=True,
                       validation_data=(X_val, y_val),
                       verbose=1)
        
        model.load_weights(file_path)
        
        oof[test_index] = np.ndarray.flatten(model.predict(X_val))
        predictions += np.ndarray.flatten(model.predict(X_s0_preds))
    
    predictions/=nfolds
    
    X_s['keras_nn'] = oof
    X_s_preds['keras_nn'] = predictions
    oof_k = np.asarray(oof)
    
    # In[Stacking]
    X_s = np.asarray(X_s)
    X_s_preds = np.asarray(X_s_preds)
    
    oof = np.zeros(len(X_s))
    predictions = np.asarray(sub.target * 0.)
    
    pipe_preprocessing = make_pipeline(
                             preprocessings['minmaxs'],
                             )
    
    full = np.concatenate((X_s, X_s_preds), axis=0)
    pipe_preprocessing.fit(full)
    X = pipe_preprocessing.transform(X_s)
    X_pred = pipe_preprocessing.transform(X_s_preds)
    #
    print('\nstart stacking... ')
    
    for fold_, (train_index, test_index) in enumerate(kf.split(X)):
        
        model = Sequential([
                Dense(512, input_shape=(X_s.shape[1],)),
                Activation('linear'),
                Dense(1),
                Activation('sigmoid'),
                ])
        
        model.compile(optimizer='adam',
                          loss='binary_crossentropy')
        
        file_path="NN_ml_"+"_model_"+"loop_"+str(fold_)+".hdf5"
        
        X_tr, X_val = X_s[train_index], X_s[test_index]
        y_tr, y_val = y[train_index], y[test_index]
        
        callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=20),
                     ModelCheckpoint(filepath=file_path, monitor='val_loss', mode='min', save_best_only=True),
                     lr_reduced]
        
        model.fit(X_tr, y_tr, 
                       epochs=750,
                       batch_size=512, 
                       callbacks=callbacks,
                       shuffle=True,
                       validation_data=(X_val, y_val),
                       verbose=1)
        
        model.load_weights(file_path)
        
        oof[test_index] = np.ndarray.flatten(model.predict(X_val))
        predictions += np.ndarray.flatten(model.predict(X_s_preds))
    
    predictions/=nfolds
    
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
        
    
    try:
        mse_score_s = mean_squared_error(y, oof_k)
        print('\rmse score - keras_nn: ', mse_score_s)   
        mse_score_s = mean_squared_error(y, oof)
        print('\rmse score - stacking: ', mse_score_s)
    except:
        pass
    
    try:
        auc_score_s = roc_auc_score(y, oof_k)
        print('auc score - keras_nn: ', auc_score_s)
        auc_score_s = roc_auc_score(y, oof)
        print('auc score - stacking: ', auc_score_s)
    except:
        pass
    
    sub['target'] = predictions
    #sub.to_csv('submission_{}.csv'.format(mse_score_s), index=False)