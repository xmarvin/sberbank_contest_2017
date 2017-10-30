from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import tqdm
import re
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

WORDS = ['как','какой','где','сколько','когда','зачем','куда','почему', 'кто', 'что', 'чей', 'кому', 'чем', 'каков','кома', 'который', 'откуда']

def find_q_word(words):
  global morph
  n_words = [morph.parse(word)[0].normal_form for word in words]
  q_words = [word for word in n_words if word in WORDS]
  if len(q_words) == 0:
    return ''  
  return q_words[0]

def uniq_words(text):
  words = re.findall("[a-zA-Zа-яёА-Я]+", text)
  words = [w.lower() for w in words]
  return list(set(words))

def get_q_word(x):  
  return find_q_word(uniq_words(x))

def make_features(df):
  for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="build features for "):        
    df.loc[index, 'qword'] = get_q_word(row['question'])
  df = df[['qword']]
  return df

train = pd.read_csv('train.csv')
train = make_features(train)
train.to_csv('strain-qword.csv', index = False)

