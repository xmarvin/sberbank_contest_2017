import os
import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.cross_validation import train_test_split
import sys
import re
import pdb
from my_model import MyModel

class MyLstm(MyModel):
  MAX_SEQUENCE_LENGTH = 30
  MAX_NB_WORDS = 200000
  EMBEDDING_DIM = 500

  num_lstm = 100#np.random.randint(175, 275)
  num_dense = 100#np.random.randint(100, 150)
  rate_drop_lstm = 0.2#0.15 + np.random.rand() * 0.25
  rate_drop_dense = 0.2#0.15 + np.random.rand() * 0.25

  def current_params(self):    
    return {}

  def uniq_words(self, text):
    words = re.findall("[a-zA-Zа-яА-Я]+", text)  
    return " ".join(list(set(words)))

  def prepare(self, train, test):
    train['paragraph'] = train['paragraph'].apply(self.uniq_words)
    train['question'] = train['question'].apply(self.uniq_words)
    test['paragraph'] = test['paragraph'].apply(self.uniq_words)
    test['question'] = test['question'].apply(self.uniq_words)

    self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS)
    self.tokenizer.fit_on_texts(pd.concat((train['paragraph'], train['question'],test['paragraph'],test['question'])))
    self.nb_words = min(self.MAX_NB_WORDS, len(self.tokenizer.word_index)) + 1    
    return (train, test)


  def get_embedding_matrix(self):
    matrix_name = 'embedding_matrix_{0}'.format(self.nb_words)
    if os.path.exists(matrix_name + '.npy'):
      embedding_matrix = np.load(matrix_name+'.npy')
    else:
      import gensim
      normal_model = gensim.models.KeyedVectors.load_word2vec_format('vendor/tenth.norm-sz500-w7-cb0-it5-min5.w2v', binary=True, unicode_errors='ignore')
      normal_model.init_sims(replace=True)
      embedding_matrix = np.zeros((self.nb_words, self.EMBEDDING_DIM))
      for word, i in self.tokenizer.word_index.items():
        if word in normal_model.wv:
          embedding_matrix[i] = normal_model.wv[word]
      np.save(matrix_name, embedding_matrix)
    return embedding_matrix

  def text_to_seq(self, text):
    sequences = self.tokenizer.texts_to_sequences(text)
    return pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)

  def nnet_model(self):
    embedding_layer = Embedding(self.nb_words,
            self.EMBEDDING_DIM,
            weights=[self.get_embedding_matrix()],
            input_length = self.MAX_SEQUENCE_LENGTH,
            trainable=False)
    
    lstm_layer = LSTM(self.num_lstm, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm)
    sequence_1_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)
    x1 = lstm_layer(x1)

    sequence_2_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)
    y1 = lstm_layer(x1)

    merged = concatenate([x1, y1])
    merged = Dropout(self.rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(self.num_dense, activation='relu')(merged)
    merged = Dropout(self.rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input], \
            outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    return model

  def train_with_params(self, x_train, x_val, y_train, y_val, nfold, params):    
    early_stopping =EarlyStopping(monitor='val_loss', patience=5)
    bst_model_path = 'tmp/lstm-{0}.h5'.format(nfold)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    model = self.nnet_model()
    model.fit(self.my_process_test(x_train), y_train, \
      validation_data=(self.my_process_test(x_val), y_val), \
      epochs=100, shuffle=True, \
      callbacks=[early_stopping, model_checkpoint])
    model.load_weights(bst_model_path)
    return model

  def my_predict_proba(self, model, x):    
    return model.predict(x)

  def my_process_test(self, x):
    x_p = self.text_to_seq(x['paragraph'])
    x_q = self.text_to_seq(x['question'])

    return [x_p, x_q]

  def my_handle_output(self,res):
    return np.array([r[0] for r in res])
