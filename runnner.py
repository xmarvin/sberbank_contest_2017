import sys
from sklearn.cross_validation import train_test_split
from save_module import *
from prepare_data import *
import json
import os.path

#from my_xgb import *
from my_lgb import *
#import my_nnet
#from my_svm import *
from my_rf import *
#from my_knn import *
#from my_ridge import *
from my_lr import *
from my_net import *
from my_rf import *
from my_lstm import MyLstm
import pdb

from my_fold import *

data_version = None

def get_data_by_level(level):
  if level == 1:
    return get_train(data_version)
  elif level == 2:
    return get_only_train_2lv(data_version)  

def load_params(path):
  if path is None:
    return None

  if not os.path.isfile(path):
    return None

  with open(path) as infile:
    data = json.load(infile)
  return data  

def run_first_level(algorithm, fname):
  train, train_y = get_data_by_level(1)
  order_idx, x_train1, y_train1, x_train2, y_train2 = get_first_split(train, train_y)
  pred2 = algorithm.predict_with_kfold(x_train1, y_train1, x_train2, 1)
  pred1 = algorithm.predict_with_kfold(x_train2, y_train2, x_train1, 1)
  res = np.concatenate((pred1, pred2), axis=0)
  res = order_split_output(order_idx, res)
  save_prediction_simple(res, fname)

def run_predict(algorithm, fname, level=1):
  if level == 1:
    train, train_y, test = get_train_and_test(data_version)
  else:
    train, train_y, test = get_train_and_test_2lv(data_version)
  res = algorithm.predict_with_kfold(train, train_y, test, 5)
  save_prediction(res, fname)

def run_cv(algorithm, level, params_path):
  train, train_y = get_data_by_level(level)
  algorithm.run_cv(train, train_y, load_params(params_path))

def hyperopt(algorithm, level):
  train, train_y = get_data_by_level(level)  
  x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=42)
  algorithm.run_hyperopt(x_train, x_valid, y_train, y_valid)

if __name__ == "__main__":
  module_name = sys.argv[1]
  algorithm = globals()[module_name]()
  algorithm_params_path = None

  if len(sys.argv) >= 4:
    data_version = sys.argv[3]
    algorithm_params_path = "config/{0}-{1}.json".format(module_name, data_version)
  if sys.argv[2] == 'predict':
    print("Predict:")
    name = "out/{0}-{1}.csv".format(module_name, data_version)
    run_predict(algorithm, name)
  elif sys.argv[2] == '1lv':
    print("1lv train:")
    name = "1lv/{0}-{1}.csv".format(module_name, data_version)
    run_first_level(algorithm, name)
  elif sys.argv[2] == 'cv2':
    run_cv(algorithm, 2, '')
  elif sys.argv[2] == 'predict2':
    name = "out/{0}-2lv.csv".format(module_name)
    run_predict(algorithm, name, 2)
  elif sys.argv[2] == 'hyper':
    print("Hyperopt:")
    hyperopt(algorithm, 1)
  elif sys.argv[2] == 'hyper2':
    print("Hyperopt - 2 level:")
    hyperopt(algorithm, 2)
  elif sys.argv[2] == 'data':
    print("Saving features:")
    save_features(data_version)
  elif sys.argv[2] == 'imp':
    train, train_y = get_data_by_level(1)
    algorithm.run_importance(train, train_y)
  else:
    print("CV:")    
    run_cv(algorithm, 1, algorithm_params_path)