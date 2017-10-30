import pandas as pd
import numpy as np
import pdb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import functools
import tqdm
import re
import os
import glob
import os.path
from nltk.stem.snowball import RussianStemmer
from nltk.corpus import stopwords
import sys
import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from prepare_data import uniq_words
VERSION = 'cluster'

normal_model = None

def wmd(s1, s2):
  global normal_model
  return normal_model.wmdistance(s1, s2)

def to_d2v(ws):
  arrs = []
  for w in ws:
    if w in normal_model.wv:
      arr = normal_model.wv[w]
      arrs.append(arr)
  if len(arrs) == 0:
    return np.zeros(500)
  return np.mean(arrs, axis=0)

def build_v_matrix(data):
  res = []
  for sentence in tqdm.tqdm(data, desc="build_v_matrix"):  
    words = uniq_words(sentence, doStem=False, doLower=True)
    vec = to_d2v(words)
    res.append(vec)  
  return res  

def save_cluster(df, train_len, cluster):  
  paragraph_data = np.load('tmp/paragraph_wmd_matrix.npy')
  question_data = np.load('tmp/question_wmd_matrix.npy')
  train = pd.DataFrame()
  test = pd.DataFrame()
  print('For cluster {0}'.format(cluster))
  p_means = MiniBatchKMeans(n_clusters=cluster, init='k-means++', init_size=10000, n_init=5, batch_size=10000, verbose=True).fit(paragraph_data)
  q_means = MiniBatchKMeans(n_clusters=cluster, init='k-means++', init_size=10000, n_init=5, batch_size=10000, verbose=True).fit(question_data)  
  train_paragraph_data = paragraph_data[:train_len]
  train_question_data = question_data[:train_len]
  train['p_cl_{0}'.format(cluster)] = p_means.predict(train_paragraph_data)
  train['q_cl_{0}'.format(cluster)] = q_means.predict(train_question_data)

  test_paragraph_data = paragraph_data[train_len:]
  test_question_data = question_data[train_len:]
  test['p_cl_{0}'.format(cluster)] = p_means.predict(test_paragraph_data)
  test['q_cl_{0}'.format(cluster)] = q_means.predict(test_question_data)

  return (train, test)

def prepare_matrix(tt_all):
  print('paragraph_data')
  paragraph_data = build_v_matrix(tt_all['paragraph'].values)
  np.save('tmp/paragraph_wmd_matrix', paragraph_data)
  print('question_data')
  question_data = build_v_matrix(tt_all['question'].values)
  np.save('tmp/question_wmd_matrix', question_data)

def save_features(cl):
  train = pd.read_csv('train.csv')  
  test = pd.read_csv('test.csv')
  tt_all = pd.concat((train,test))
  tt_all = tt_all[['paragraph', 'question']]
    
  if cl is None:
    import gensim
    global normal_model
    normal_model = gensim.models.KeyedVectors.load_word2vec_format('vendor/tenth.norm-sz500-w7-cb0-it5-min5.w2v', binary=True, unicode_errors='ignore')
    normal_model.init_sims(replace=True)
    prepare_matrix(tt_all)
  else:
    train, test = save_cluster(tt_all, train.shape[0],cl)
    train.to_csv('strain-{0}-{1}.csv'.format(VERSION, cl), index=False)
    test.to_csv('stest-{0}-{1}.csv'.format(VERSION, cl), index=False)


cl = None
if len(sys.argv) > 1:
  cl = int(sys.argv[1])
save_features(cl)
