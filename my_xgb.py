import pandas as pd
import numpy as np
import operator
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from operator import add
from sklearn.feature_selection import VarianceThreshold
import pdb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.decomposition import PCA
import sys

from my_model import MyModel

class MyXgbModel(MyModel):

  def current_params(self):
    params = {
      'objective': 'binary:logistic',
      'eval_metric': 'auc',
      'colsample_bytree': 0.6,
      'eta': 0.02,
      'gamma': 0.7,
      'subsample': 0.6,
      'max_depth': 3,
      'silent': 1,
      'min_child_weight': 5.0,
      'tree_method': 'exact'
    }
    return(params)

  def train_with_params(self, x_train, x_valid, y_train, y_valid, nfold, params):
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=30, verbose_eval=100)
    return bst

  def my_predict_proba(self, bst, d_test):  
    p_test = bst.predict(d_test, ntree_limit=bst.best_iteration)
    return p_test

  def my_process_test(self,t):
    return xgb.DMatrix(t)

  def my_handle_output(self, res):
    return res

  def hyperopt_scope(self):
    space = {      
        'eta': hp.quniform('eta', 0.005, 0.1, 0.025),
        'max_depth':  hp.choice('max_depth', [4,5,6,7]),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',      
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': 0
    }
    return space