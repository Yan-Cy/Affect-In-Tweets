# coding: utf-8
# pylint: disable = invalid-name, C0111
import data_helper
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import neighbors
import xgboost as xgb

tasks = ['anger', 'fear', 'joy', 'sadness']
gensim_model = None

for task in tasks:
    print 'Running for task', task
    # load or create your dataset
    print('Load data...')
    X_train, y_train, train_id, train_raw, gensim_model = data_helper.load_dataset('train', task, gensim_model)
    X_test, y_test, test_id, test_raw, gensim_model = data_helper.load_dataset('test', task, gensim_model)

    
    #--------------------------------- Lightgbm--------------------------------------
    
    # create dataset for lightgbm
    #print X_train, y_train
    #print type(X_train), type(y_train)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    '''
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    '''

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 10,
        'num_rounds': 200,
    }
    params2 = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 10,
        'num_rounds': 200,
    }

    print('Start training...')
    # train
    #gbm = lgb.train(params,
    #                lgb_train,
    #                num_boost_round=200,
    #                valid_sets=lgb_eval,
    #                early_stopping_rounds=5)
    gbm1 = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    verbose_eval=5)
    gbm2 = lgb.train(params2,
                    lgb_train,
                    valid_sets=lgb_eval,
                    verbose_eval=5)
    #print('Save model...')
    # save model to file
    #gbm.save_model('model.txt')

    print('Start predicting...')
    # predict
    y_pred1 = gbm1.predict(X_test, num_iteration=gbm1.best_iteration)
    y_pred2 = gbm2.predict(X_test, num_iteration=gbm2.best_iteration)
    y_pred = np.mean([y_pred1, y_pred2], axis = 0)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    
    
    #--------------------------------- xgboost -------------------------------------
    #xgb_model = xgb.XGBRegressor().fit(X_train, y_train)
    #y_pred = xgb_model.predict(X_test)
    #print(mean_squared_error(y_test, y_pred) ** 0.5)


    #--------------------------------- Linear Regression -------------------------
    '''
    ridge = linear_model.Ridge (alpha = .5)
    y_rid = ridge.fit(X_train, y_train).predict(X_test)

    bayesian = linear_model.BayesianRidge()
    y_bay = bayesian.fit(X_train, y_train).predict(X_test)

    lar = linear_model.Lars(n_nonzero_coefs=1)
    y_lar = lar.fit(X_train, y_train).predict(X_test)


    n_neighbors = 30
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    y_knn = knn.fit(X_train, y_train).predict(X_test)


    print 'Ridge Regression', mean_squared_error(y_test, y_rid) ** 0.5
    print 'Bayesian Ridge Regression', mean_squared_error(y_test, y_bay) ** 0.5
    print 'Least Angle Regression', mean_squared_error(y_test, y_lar) ** 0.5
    print 'KNN Regression', mean_squared_error(y_test, y_knn) ** 0.5
    '''

    #--------------------------------- Support Vector Regression -------------------
    # Fit regression model
    '''
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
    y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
    y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
    print 'SVR rbf', mean_squared_error(y_test, y_rbf) ** 0.5
    print 'SVR lin', mean_squared_error(y_test, y_lin) ** 0.5
    print 'SVR poly', mean_squared_error(y_test, y_poly) ** 0.5
    '''

    #--------------------------------- Output --------------------------------------
    output = np.column_stack((test_id, test_raw, [task] * len(test_id), y_pred))
    out_path = '../results/{}-pred.txt'.format(task)
    with open(out_path, 'w') as f:
        for x in output:
            f.write('\t'.join(x) + '\n')
