import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cross_validation import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from my_fold import *
import pdb

class MyModel:
  def current_params(self):  
    raise Exception('Imlement me')

  def prepare(self, train, test):
    return (train, test)

  def my_train(self, x_train, x_val, y_train, y_val, nfold, params = None):
    if params == None:
      params = self.current_params()

    return self.train_with_params(x_train, x_val, y_train, y_val, nfold, params)

  def train_with_params(self, x_train, x_val, y_train, y_val, nfold, params):
    return None

  def my_predict_proba(self,model, x):
    return model.predict(x)

  def my_process_test(self,t):  
    return t

  def my_handle_output(self,res):
    return res

  def predict_with_kfold(self, x_train, y_train, x_valid, nfolds, params = None):  
    if params == None:
      params = self.current_params()
    print(params)
    return predict_kfold(self, x_train, y_train, x_valid, nfolds, params)

  def hyperopt_score(self, params):
    y_pred = self.predict_with_kfold(self.x_train, self.y_train, self.x_valid, 1, params)
    return roc_auc_score(self.y_valid, y_pred)    

  def hyperopt_scope(self):
    raise Exception('Imlement me')

  def run_importance(self, train, train_y, params = None):
    if params == None:
      params = self.current_params()
    x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=42)
    self.show_importance(x_train, x_valid, y_train, y_valid, params)

  def get_index(arr, val):
    for position, item in enumerate(arr):
      if item == val:
        return position
    return None

  def run_cv(self, train, train_y, params = None):
    print(train.head())
    n = 3
    scores = []
    accuracy = []
    for i in range(n):
      train['index'] = range(0,train.shape[0])
      x_train, x_valid, y_train, y_valid = train_test_split(train, train_y, test_size=0.2, random_state=43+i)
      valid_index = x_valid['index'].values
      x_train.drop('index', axis=1,inplace=True)
      x_valid.drop('index', axis=1,inplace=True)
      y_pred = self.predict_with_kfold(x_train, y_train, x_valid, 5, params)
      y_pred_int = [int(a>=0.5) for a in y_pred]
      #pdb.set_trace()
      scores.append(roc_auc_score(y_valid, y_pred))
      accuracy.append(accuracy_score(y_valid, y_pred_int))
    scores = np.array(scores)
    accuracy = np.array(accuracy)
    print("ROC: min = {0}, max = {1}, avg = {2}".format(scores.min(), scores.max(), scores.mean()))
    print("Accuracy: min = {0}, max = {1}, avg = {2}".format(accuracy.min(), accuracy.max(),accuracy.mean()))

  def run_hyperopt(self, x_train, x_valid, y_train, y_valid):
    self.x_train = x_train
    self.x_valid = x_valid
    self.y_train = y_train
    self.y_valid = y_valid
    best = fmin(self.hyperopt_score, self.hyperopt_scope(),
      algo=tpe.suggest, max_evals=1000)
    self.x_train = None
    self.x_valid = None
    self.y_train = None
    self.y_valid = None
    print(best)    