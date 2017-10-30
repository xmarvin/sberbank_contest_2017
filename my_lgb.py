import pandas as pd
import numpy as np
import operator
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from operator import add
from sklearn.feature_selection import VarianceThreshold
import pdb
from sklearn.decomposition import PCA
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import sys

from my_model import MyModel

class MyLgbModel(MyModel):

  def current_params(self):
    params = {
      'task': 'train',
      'boosting_type': 'gbdt',
      'objective': 'binary',
      'metric': 'auc',
      'bagging_freq': 10,
      'num_leaves': 10,
      'bagging_fraction': 0.75,
      'feature_fraction': 0.6,
      'learning_rate': 0.025,
      'lambda_l2': 1e-3,
      'verbose': 0
    }
    return(params)

  def train_with_params(self,x_train, x_valid, y_train, y_valid, nfold, params):
    d_train = lgb.Dataset(x_train.values, y_train)
    d_valid = lgb.Dataset(x_valid.values, y_valid, reference=d_train)  
    bst = lgb.train(params, d_train, num_boost_round = 10000, valid_sets=d_valid, early_stopping_rounds=10)
    return bst

  def show_importance(self, x_train, x_valid, y_train, y_valid, params):
    bst = self.train_with_params(x_train, x_valid, y_train, y_valid, 0, params)
    features = x_train.columns
    importance = sorted(zip(features, bst.feature_importance()), key=lambda x: x[1], reverse=True)
    for imp in importance[:30]: print("Feature '{}', importance={}".format(*imp))

  def my_predict_proba(self,bst, d_test):  
    p_test = bst.predict(d_test, num_iteration=bst.best_iteration)
    return p_test

  def my_process_test(self,t):
    return t.values

  def hyperopt_scope(self):
    space = {
      'task': 'train',
      'boosting_type': 'gbdt',
      'objective': 'binary',
      'metric': 'auc',
      'num_leaves': hp.choice('num_leaves', [10,20,25,30,31,35,40,45,50]),
      'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
      'feature_fraction': hp.quniform('feature_fraction', 0.5, 1, 0.05),
      'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1, 0.05),
      'bagging_freq': hp.choice('bagging_freq', [3,4,5,6,7,8,10])
    }
    return space