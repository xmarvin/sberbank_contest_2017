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

idfs = None
stemer = RussianStemmer()

def uniq_words(text, doStem=True, doLower=False, doStop = False, doUniq = True):
  words = re.findall("[a-zA-Zа-яёА-Я]{2,}", text)
  if doLower:
    words = [a.lower() for a in words]
  if doStop:
    words = [word for word in words if word not in stopwords.words('russian')]
  if doStem:
    words = [stemer.stem(word) for word in words]
  if doUniq:
    words = np.unique(words)
  return words

def calculate_idfs(data):
    counter_paragraph = Counter()
    uniq_paragraphs = data['paragraph'].unique()
    for paragraph in tqdm.tqdm(uniq_paragraphs, desc="calc idf"):
        set_words = uniq_words(paragraph, doStem = True)
        counter_paragraph.update(set_words)
        
    num_docs = uniq_paragraphs.shape[0]
    idfs = {}
    for word in counter_paragraph:
        idfs[word] = np.log(num_docs / counter_paragraph[word])
    return idfs

def make_features(df):
  global idfs

  df.drop('paragraph_id', axis=1,inplace=True)
  df.drop('question_id', axis=1,inplace=True)

  for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="build features for "):
    question = uniq_words(row.question, doStem = True)
    paragraph = uniq_words(row.paragraph, doStem = True)
    df.loc[index, 'len_paragraph'] = len(paragraph)
    df.loc[index, 'len_question'] = len(question)
    df.loc[index, 'len_intersection'] = len(np.intersect1d(paragraph, question))
    df.loc[index, 'idf_question'] = np.sum([idfs.get(word, 0.0) for word in question])
    df.loc[index, 'idf_paragraph'] = np.sum([idfs.get(word, 0.0) for word in paragraph])
    df.loc[index, 'idf_intersection'] = np.sum([idfs.get(word, 0.0) for word in np.intersect1d(paragraph, question)])

    sent_len_intersection = []
    sent_idf_intersection = []
    paragraph_arr = row.paragraph.split('.')
    for psent in paragraph_arr:    
      psent_tokens = uniq_words(psent, doStem = True)
      if len(psent_tokens) > 2: 
        sent_len_intersection.append(len(np.intersect1d(psent_tokens, question)))
        sent_idf_intersection.append(np.sum([idfs.get(word, 0.0) for word in np.intersect1d(psent_tokens, question)]))

    df.loc[index, 'max_sent_len_int'] = np.max(sent_len_intersection)
    df.loc[index, 'max_sent_idf_int'] = np.max(sent_idf_intersection)


  df.drop('question', axis=1,inplace=True)
  df.drop('paragraph', axis=1,inplace=True)
  return(df)

def save_features(version):
  global idfs
  train = pd.read_csv('train.csv')
  test = pd.read_csv('test.csv')
  tt_all = pd.concat((train,test))
  tt_all = tt_all[['paragraph', 'question']]  

  idfs = calculate_idfs(tt_all)  
  train = make_features(train)
  train.to_csv('strain-{0}.csv'.format(version), index=False)
  test = make_features(test)
  test.to_csv('stest-{0}.csv'.format(version), index=False)

def get_train_and_test(data_version):
  train, train_y = get_train(data_version)
  test = get_test(data_version)
  return (train, train_y, test)

def add_paragraph_count(df):
  h = dict(df['paragraph_id'].value_counts())
  df['paragraph_count'] = df['paragraph_id'].apply(lambda x: h[x])
  return df

def save_y():
  train = pd.read_csv('train.csv')
  train = train[['target']]
  train.to_csv('train_y.csv', index = False)

def get_train(data_version):
  path = 'train.csv'
  if data_version is None:
    train = pd.read_csv('train.csv')
  else:
    path = 'strain-{0}.csv'.format(data_version)

  train = pd.read_csv(path)

  if 'target' in train.columns:
    train_y = train['target'].values.astype(int)
    train.drop('target', axis=1, inplace=True)    
  else:
    train = pd.read_csv(path)
    train_y = pd.read_csv('train_y.csv')
    train_y = train_y['target'].values.astype(int)

  return (train, train_y)

def get_test(data_version):
  path = 'test.csv'
  if not data_version is None:
    path = 'stest-{0}.csv'.format(data_version)
  test = pd.read_csv(path)  
  return test

def get_train_and_test_2lv(data_version):
  train, train_y = get_only_train_2lv(data_version)
  test = get_test_2lv(data_version)
  return (train, train_y, test)

def get_test_2lv(data_version):
  test = pd.DataFrame()
  for fname in glob.glob("out1/*.csv"):
    name = os.path.basename(fname).split('.')[0]
    fl = pd.read_csv(fname)
    test[name] = fl['prediction']

  return test  

def get_only_train_2lv(data_version):
  train_df, train_y = get_train(data_version)
  train = pd.DataFrame()
  for fname in glob.glob("1lv/*.csv"):
    name = os.path.basename(fname).split('.')[0]
    fl = pd.read_csv(fname)
    train[name] = fl['prediction']

  return (train, train_y)

def append_column_from(in_paths, out_path, columns):  
  out_df = pd.read_csv(out_path)
  for in_path in in_paths:
    in_df = pd.read_csv(in_path)     
    if columns == '*':
      cols = in_df.columns
    else:
      cols = columns  
    for column in cols:
      out_df[column] = in_df[column]
  out_df.to_csv(out_path, index=False)

def remove_column_from(out_path, columns):
  out_df = pd.read_csv(out_path)  
  cols = [column for column in out_df.columns if column not in columns]
  out_df = out_df[cols]
  out_df.to_csv(out_path, index=False)