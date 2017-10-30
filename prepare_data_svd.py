from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import pymorphy2
from sklearn.decomposition import TruncatedSVD

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def save_sparse_csr(filename,array):
  np.savez(filename,data = array.data ,indices=array.indices,
              indptr =array.indptr, shape=array.shape) 
def load_sparse_csr(filename):
  loader = np.load(filename)
  return csr_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def csr_to_svd(data, com, th):
  svd = TruncatedSVD(n_components=com, n_iter=20, random_state=42)
  z = np.squeeze(np.asarray(data.sum(axis=0) > th))  
  return svd.fit_transform(data[:,z])

def build_df(pa, qa):
  t1 = pd.DataFrame(pa)
  t2 = pd.DataFrame(qa)
  t1.columns = ['p_{0}'.format(a)  for a in t1.columns]
  t2.columns = ['q_{0}'.format(a)  for a in t2.columns]
  tt = t1.join(t2)
  return tt

text = np.concatenate(( (train.paragraph.values + train.question.values), (test.paragraph.values + test.question.values)))
morph = pymorphy2.MorphAnalyzer()
cv = CountVectorizer(tokenizer= lambda x: [morph.parse(a)[0].normal_form for a in re.findall("[a-zA-Zа-яА-Я]+", x)] )
cv.fit(text)

train_paragraph_m = cv.transform(train['paragraph'])
train_paragraph_m = csr_to_svd(train_paragraph_m, 500, 5)
train_question_m = cv.transform(train['question'])
train_question_m = csr_to_svd(train_question_m, 50, 5)
test_paragraph_m = cv.transform(test['paragraph'])
test_paragraph_m = csr_to_svd(test_paragraph_m, 500, 5)
test_question_m = cv.transform(test['question'])
test_question_m = csr_to_svd(test_question_m, 50, 5)

train2 = build_df(train_paragraph_m, train_question_m)
train2['target'] = train['target']
train2.to_csv('strain-svd.csv')
test2 = build_df(test_paragraph_m, test_question_m)
test2.to_csv('stest-svd.csv')


#save_sparse_csr('train_paragraph.csr', paragraph_m)
#save_sparse_csr('train_question.csr',  question_m)
#question_m = load_sparse_csr('train_question.csr.npz')
#paragraph_m = load_sparse_csr('train_paragraph.csr.npz')
