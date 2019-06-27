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
Ver: 01.02.01 Modulation
Ver: 01.02.02 Autoencoder


-----------------------------------
TODO: 0. Add xgboost sklearn 
TODO: 1. selectable metric as metric 3 (custom)
TODO: 2. using Keras NN to stack the models
@author: ML
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model

import numpy as np
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

def auto_model(X, y, X_pred, sub, nfolds=5):
    
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
       
    lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.01,
                                   patience=5,
                                   verbose=0,
                                   min_delta=1e-5,
                                   mode='min')
    
    def pre_process(X, X_pred, mode='full'):
             
        preprocessings = {
                'standards':preprocessing.StandardScaler(),
                'minmaxs':preprocessing.MinMaxScaler(),
                'robusts':preprocessing.RobustScaler(),
                'PCA': PCA(),
                'PowerTransformer':preprocessing.Normalizer(), 
                'variance_threshold':feature_selection.VarianceThreshold(threshold=0.5),        
                      }
        
        if mode == 'full':
            pipe_preprocessing = make_pipeline(
            #                        preprocessings['variance_threshold'],
                                     preprocessings['standards'],
    #                                 preprocessings['minmaxs'],
                                     preprocessings['robusts'],
    #                                 preprocessings['PCA'],
                                     preprocessings['PowerTransformer'],                               
                                    )
        elif mode == 'min_max_scale':
            pipe_preprocessing = make_pipeline(
                    preprocessings['minmaxs'],
                    preprocessings['PCA'],
                    )
            
        full = np.concatenate((X, X_pred), axis=0)
        pipe_preprocessing.fit(full)
        X = pipe_preprocessing.transform(X)
        X_pred = pipe_preprocessing.transform(X_pred)
        
        return X, X_pred

  
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
    
    from sklearn.model_selection import KFold    
    kf = KFold(n_splits=nfolds, random_state=2019, shuffle=True)

    def auto_encoding(X):
        
        input_img = Input(shape=(X.shape[1],))
        encoded = Dense(20, activation='relu')(input_img)
        decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(X, X,
                        epochs=20,
                        batch_size=32,
                        shuffle=True,
                        validation_split=0.8)
        
        return autoencoder.predict(X)

    models = {
            'ridge  ': linear_model.Ridge(alpha=0.1, normalize= False),
#            'lasso  ': linear_model.Lasso(alpha=1e-6, max_iter=1e8),
            'lr     ': linear_model.LogisticRegression(solver='lbfgs',warm_start=True, max_iter=1e4),
#            'lrCV   ': linear_model.LogisticRegressionCV(solver='lbfgs', max_iter=1e4, cv=5),
            'mlp_clf': neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(256, 64), random_state=1),
            'mlp_reg': neural_network.MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(256, 64, 32, 32, 32), random_state=1),
            'svc    ': svm.SVC(C= 10, kernel='rbf'),
            'rfreg  ': ensemble.RandomForestRegressor(max_depth=15),
            'rfclf  ': ensemble.RandomForestClassifier(max_depth=12),
            'lgbclf ': lgb.LGBMClassifier(gamma='auto', num_leaves=4,learning_rate=0.001, n_estimators=2000, verbose = 100),
            'lgbreg ': lgb.LGBMRegressor(gamma='auto', num_leaves=31,learning_rate=0.001, n_estimators=20000, verbose = 100),
            'knn    ': neighbors.KNeighborsClassifier(n_neighbors=30, n_jobs=15),
            'nb     ': naive_bayes.GaussianNB(),
            'dt     ': tree.DecisionTreeClassifier(),
            'catreg ': CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=3, verbose = 100),
            'catclf ': CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=2, verbose = 100),
            }
    
    print('\nall models: ', list(models.keys()))

    X, X_pred = pre_process(X, X_pred, 'full')
    
    # In[Auto_encoding]
    
    #
    mse_score = []
    auc_score = []
    valid_score = {}
    oof = y*0
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
                            early_stopping_rounds=200,
                            ) 
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

    full = np.concatenate((X, X_pred), axis=0)
    full = auto_encoding(full)
    X = full[:len(X)]
    X_pred = full[len(X):]    
    
    X_s0 = np.asarray(X)
    X_s0_preds = np.asarray(X_pred)
    
    oof = np.zeros(len(X_s0))
    predictions = np.asarray(sub.target * 0.)
    
    X, X_pred = pre_process(X_s0, X_s0_preds, 'min_max_scale')
    #
    
    print('\nstart kerasNN ... ')
    
    for fold_, (train_index, test_index) in enumerate(kf.split(X)):
        
        model = Sequential([
                Reshape((X.shape[1], 1, 1,), input_shape=(X.shape[1],)),
                Dense(512),
                Activation('relu'),
                Dense(128),
                Activation('relu'),
                Dense(64),
                Activation('relu'),
                Dense(64),
                Activation('relu'),
                Flatten(),
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
    
    X, X_pred = pre_process(X_s, X_s_preds, 'min_max_scale')
    #
    print('\nstart stacking... ')
    
    for fold_, (train_index, test_index) in enumerate(kf.split(X)):
        
        model = Sequential([
                Reshape((X.shape[1], 1, 1, ), input_shape=(X.shape[1],)),
                Dropout(0.2),
                Dense(512),
                Activation('relu'),
                Dense(64),
                Activation('linear'),
                Dense(64),
                Activation('relu'),
                Dense(64),
                Activation('linear'),
                Flatten(),
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
                       batch_size=64, 
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
    
    try:
        inds = np.argmax(mse_score) 
        print('\nmse score - worst model found: ',  list(models.keys())[inds], np.max(mse_score))
        inds = np.argmin(auc_score) 
        print('auc score - worst model found: ',  list(models.keys())[inds], np.min(auc_score))
        
        inds = np.argmin(mse_score) 
        print('mse score - best model found: ',  list(models.keys())[inds], np.min(mse_score))
        inds = np.argmax(auc_score) 
        print('auc score - best model found: ',  list(models.keys())[inds], np.max(auc_score))
    except:
        pass
            
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
    sub.to_csv('submission_{}.csv'.format(mse_score_s), index=False)
    
    return

'''
auc score - keras_nn:  0.9927461550657999
auc score - stacking:  0.9931293272025792


with auto encoding
auc score - keras_nn:  0.9588354209608372
auc score - stacking:  0.9931425400348819
'''