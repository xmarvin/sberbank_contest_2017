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

def bigrams(x):
  arr = []
  for px in x['paragraph'].split(' '):
    for qx in x['question'].split(' '):
      arr.append('{0}_{1}'.format(px, qx))

  return ' '.join(list(set(arr)))

def uniq_words(text):
  words = re.findall("[a-zA-Zа-яА-Я]+", text)
  words = [w.lower() for w in words]
  return " ".join(list(set(words)))

train = pd.read_csv('train.csv')
train = train[train.target==1]

train['paragraph'] = train['paragraph'].apply(uniq_words)
train['question'] = train['question'].apply(uniq_words)

train['bigrams'] = train.apply(bigrams, axis=1, raw=True)
train = train[['bigrams']]

train.to_csv('strain-positive.csv', index = False)

