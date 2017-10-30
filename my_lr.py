import pandas as pd
import numpy as np
import operator
from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from operator import add
from sklearn.feature_selection import VarianceThreshold
import pdb
from sklearn.decomposition import PCA
import sys

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from my_model import MyModel

class MyLrModel(MyModel):

  def current_params(self):
    res = {    
      'normalize': False,
      'fit_intercept': True
    }
    return res

  def train_with_params(self, x_train, x_val, y_train, y_val, nfold, params):   
    clf = LinearRegression(**params)
    clf.fit(x_train, y_train)
    return clf

  def my_predict_proba(self,model, x):
    return model.predict(x)

  def hyperopt_space(self):
    space = {       
         'normalize': hp.choice('normalize', [True, False]),
         'fit_intercept': hp.choice('fit_intercept', [True, False])
         }
    return space

    