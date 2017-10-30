from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import tqdm
from prepare_data import uniq_words
import re


def find_trigrams(input_list):  
  return [' '.join(a) for a in zip(input_list, input_list[1:], input_list[2:])]

def find_bigrams(input_list):
  return [' '.join(a) for a in zip(input_list, input_list[1:])]

def bigrams_int(p,q):
  p_grams = find_bigrams(p)
  q_grams = find_bigrams(q)
  return len(np.intersect1d(p_grams,q_grams))

def trigrams_int(p,q):
  p_grams = find_trigrams(p)
  q_grams = find_trigrams(q)
  return len(np.intersect1d(p_grams,q_grams))

def make_features(df):
  for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="build features for "):    
    p_tokens = uniq_words(x['paragraph'], doUniq=False, doLower = True, doStem = True, doStop = False)
    q_tokens = uniq_words(x['question'], doUniq=False, doLower = True, doStem = True, doStop = False)

    df.loc[index, 'bigrams_int'] = bigrams_int(row)
    df.loc[index, 'trigrams_int'] = trigrams_int(row)  
  df = df[['bigrams_int', 'trigrams_int']]
  return df

train = pd.read_csv('train.csv')
train = make_features(train)
train.to_csv('strain-intergrams.csv', index = False)

test = pd.read_csv('test.csv')
test = make_features(test)
test.to_csv('stest-intergrams.csv', index = False)