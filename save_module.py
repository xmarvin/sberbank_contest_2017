import pandas as pd
import numpy as np

def save_prediction(p1, name):
  test = pd.read_csv('test.csv')
  test = test[['paragraph_id','question_id']]
  test['prediction'] = p1
  test.to_csv(name, index=False)

def save_prediction_simple(p1, name):
  sub = pd.DataFrame()
  sub['prediction'] = p1
  sub.to_csv(name, index=False)