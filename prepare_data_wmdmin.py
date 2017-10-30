import gensim
import numpy as np
import pandas as pd
import re
import tqdm
import os
import csv
from sklearn.metrics.pairwise import cosine_similarity
from prepare_data import uniq_words

normal_model = gensim.models.KeyedVectors.load_word2vec_format('vendor/tenth.norm-sz500-w7-cb0-it5-min5.w2v', binary=True, unicode_errors='ignore')
normal_model.init_sims(replace=True)

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

def save_matrix_features(df, fname):
  arrs = []
  for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="build features for "):
    wmds = []
    wcos = []
    question = list(uniq_words(row.question, doLower=True, doStem = False))
    arrq = to_d2v(question)
    paragraph_arr = row.paragraph.split('.')
    for psent in paragraph_arr:    
      paragraph = list(uniq_words(psent, doLower=True, doStem = False))
      if len(paragraph) > 2: 
        wmds.append(wmd(paragraph,question))

        arrp = to_d2v(paragraph)
        wcos.append(cosine_similarity(arrp.reshape(1,-1),arrq.reshape(1,-1))[0][0])  

    arrs.append([np.min(wmds),np.min(wcos)])

  ds = pd.DataFrame(arrs, columns=['wmdmin','cosmin'])
  ds.to_csv(fname, index = False)

train = pd.read_csv('train.csv')
save_matrix_features(train, 'strain-wmd.csv')
test = pd.read_csv('test.csv')
save_matrix_features(test, 'stest-wmd.csv')


