import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from operator import add

def order_split_output(order_idx, x):  
  z = [a for a in zip(order_idx, x) ]
  z.sort(key=lambda x: x[0])
  return [z1[1] for z1 in z]

def get_first_split(train, train_y):
  kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
  train_index, test_index = [train_index for i, (train_index, test_index) in enumerate(kf.split(train, train_y))]

  x_train1, x_train2 = train.iloc[train_index], train.iloc[test_index]
  y_train1, y_train2 = train_y[train_index], train_y[test_index]

  idx = list(train_index) + list(test_index)
  return(idx, x_train1, y_train1, x_train2, y_train2)

def get_first_split_simple(train, train_y):
  mid = train.shape[0] // 2

  x_train1 = train[:mid]
  y_train1 = train_y[:mid]
  x_train2 = train[mid:]
  y_train2 = train_y[mid:]

  return(x_train1, y_train1, x_train2, y_train2)

def predict_kfold(module, train, train_y, test, folds,prs=None):
  nfolds = folds
  if nfolds == 1:
    nfolds = 5
  kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
  pred = None
  train, test = module.prepare(train, test)  
  dtest = module.my_process_test(test)
  for i, (train_index, test_index) in enumerate(kf.split(train, train_y)):
    x_train, x_valid = train.iloc[train_index], train.iloc[test_index]
    y_train, y_valid = train_y[train_index], train_y[test_index]
    bst = module.my_train(x_train, x_valid, y_train, y_valid, i,prs)
    y_pred = module.my_predict_proba(bst, dtest)
    if pred is None:
      if folds == 1:
        return module.my_handle_output(y_pred)

      pred = [y_pred]      
    else:
      pred = np.vstack((pred,[y_pred]))

  res = pred.mean(axis=0)
  return module.my_handle_output(res)