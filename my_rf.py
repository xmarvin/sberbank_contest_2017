import pandas as pd
import numpy as np
import operator
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from operator import add
from sklearn.feature_selection import VarianceThreshold
import pdb
from sklearn.decomposition import PCA

import sys

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.grid_search import GridSearchCV

from my_model import MyModel

class MyRfModel(MyModel):

  def current_params(self):
    params = {       
         'n_estimators': 100,
         'max_features': 'auto',
         'min_samples_split': 3,
         'min_samples_leaf': 2,
         'max_leaf_nodes': 100,
         'n_jobs': -1
         }
    return(params)

  def train_with_params(self, x_train, x_val, y_train, y_val, nfold, params):
    params['random_state'] = nfold
    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)
    return clf

  def my_predict_proba(self, model, x):
    return model.predict_proba(x)

  def my_process_test(self, t):  
    return t

  def my_handle_output(self, res):
    return np.array([r[1] for r in res])

  def run_hyperopt(self, x_train, x_valid, y_train, y_valid):
    space = {       
         'n_estimators': [1000,1200,1500,1800,2000],
         'max_features': ['log2', 'auto', 'sqrt'],
         'min_samples_split': [2,3,4,5],
         'min_samples_leaf': [2,3,4,5],
         'max_leaf_nodes': [50,60,75,100],
         }      
    clf = RandomForestClassifier({'random_state': 0})
    grid_search = GridSearchCV(clf, space, n_jobs=-1, cv=2)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
