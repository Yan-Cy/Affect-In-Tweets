# coding: utf-8
# pylint: disable = invalid-name, C0111
import data_helper
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

tasks = ['anger', 'fear', 'joy', 'sadness']
gensim_model = None

for task in tasks:
    print 'Running for task', task
    # load or create your dataset
    print('Load data...')
    X_train, y_train, train_id, train_raw, gensim_model = data_helper.load_dataset('train', task, gensim_model)
    X_test, y_test, test_id, test_raw, gensim_model = data_helper.load_dataset('test', task, gensim_model)

    # create dataset for lightgbm
    #print X_train, y_train
    #print type(X_train), type(y_train)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict

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
        'num_rounds': 500,
    }
    '''

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,
                    valid_sets=lgb_eval)
    #                early_stopping_rounds=5)
    #gbm = lgb.train(params,
    #                lgb_train,
    #                valid_sets=lgb_eval,
    #                verbose_eval=5)

    print('Save model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


    output = np.column_stack((test_id, test_raw, [task] * len(test_id), y_pred))
    out_path = '../results/{}-pred.txt'.format(task)
    with open(out_path, 'w') as f:
        for x in output:
            f.write('\t'.join(x) + '\n')
