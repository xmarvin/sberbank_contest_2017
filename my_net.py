import pandas as pd
import numpy as np

import operator
import sys
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from operator import add
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import pdb
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Conv1D, MaxPool1D, Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from save_module import *
from prepare_data import *
from my_fold import *
import random

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

random.seed(0)

from my_model import MyModel

class MyNetModel(MyModel):

  def current_params(self):  
    {}

  def my_process_test(self, t):  
    #return t.as_matrix()    
    return t.as_matrix().reshape(t.shape[0], t.shape[1],1) #lstm
  def hyperopt_scope(self):
    return None

  def model(self,x_train, x_valid, y_train, y_valid):
    model = Sequential()
    model.add(Dense({{choice([100,200,300,400,500])}}, input_dim=16, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2759433172288843))
    model.add(Dense({{choice([100,200,300,400,500])}}, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.32))
    model.add(Dense({{choice([100,200,300,400,500])}}, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer={{choice(['rmsprop', 'adadelta', 'sgd'])}}, metrics=['accuracy'])
    model.fit(x_train, y_train,
      batch_size={{choice([64, 128, 256,512,1024])}},
      epochs=10,
      verbose=2,
      validation_data=(x_valid, y_valid))

    score, acc = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

  def run_hyperopt(self, x_train, x_valid, y_train, y_valid):
      best_run, best_model = optim.minimize(model=model,
                                            data=(x_train, x_valid, y_train, y_valid),
                                            algo=tpe.suggest,
                                            max_evals=5,
                                            rseed=0,
                                            trials=Trials())
   
      print("Evalutation of best performing model:")
      print(best_model.evaluate(x_valid, y_valid))
      print("Best performing model chosen hyper-parameters:")
      print(best_run)

  def nnet_model(self,dim):
    model = Sequential()
    model.add(Dense(1000, input_dim=dim, kernel_initializer='normal', activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

  def nnet_model_conv(self,dim):
    model = Sequential()
    model.add(Conv1D(2,2,input_shape=(dim,1,),kernel_initializer= 'uniform'))
    model.add(MaxPool1D(2))
    model.add(Conv1D(2,2, kernel_initializer= 'uniform'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(500,  activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model  

  def train_with_params(self,x_train, x_val, y_train, y_val, nfold,prs=None):
    bst_model_path = "nnet_weights-{0}.h5".format(nfold)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    model = self.nnet_model_conv(x_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(self.my_process_test(x_train), y_train, validation_data=(self.my_process_test(x_val), y_val),
      shuffle=True, batch_size=128, callbacks=[early_stopping, model_checkpoint], epochs=200)
    model.load_weights(bst_model_path)
    return model

  def my_predict_proba(self,model, x):
    return model.predict_proba(x)

  def my_handle_output(self,res):
    return np.array([r[0] for r in res])
  