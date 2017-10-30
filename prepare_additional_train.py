import pandas as pd
import numpy as np
import tqdm
import re
import os
import random
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
ALLOWED_TAGS = ['NOUN', 'PREP', 'PRCL', 'INTJ', 'NPRO', 'NUMR', 'UNKN']

dictionary = set()

def uniq_words(text):
  words = re.findall("[a-zA-Zа-яёА-Я]{2,}", text)
  words = [w.lower() for w in words]
  return list(set(words))

def get_tag(x):
  return str(morph.tag(x)[0]).split(',')[0]

def remove_sense(x):
  global dictionary
  tokens = uniq_words(x)
  allowed_tokens = [t for t in tokens if get_tag(t) in ALLOWED_TAGS]
  missing_count = len(tokens) - len(allowed_tokens)
  allowed_tokens = np.append(allowed_tokens, random.sample(dictionary, missing_count))
  random.shuffle(allowed_tokens)
  return ' '.join(allowed_tokens)

train = pd.read_csv('train.csv')
for question in train.question.values:
  for token in uniq_words(question):
    dictionary.add(token)

pos_train = train[train.target == 1]
pos_train['target'] = 0
pos_train['question'] = pos_train['question'].apply(remove_sense)
first_q = np.max(pos_train['question_id']) + 1
pos_train['question_id'] = [ int(a) for a in np.linspace(first_q, first_q + pos_train.shape[0] - 1,  pos_train.shape[0])]
train = train.append(pos_train)
train.to_csv('train2.csv', index=False)

