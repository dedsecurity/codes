#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import random
import nltk
import torch
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM , Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt

print(tf.__version__)

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.config.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

"""
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU {}'.format(tpu.cluster_spec().as_dict()['worker']))
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()
print("REPLICAS: {}".format(strategy.num_replicas_in_sync))
"""

class Nl2PyTranslator(tf.keras.Model):
    def __init__(self, nl_text_processor, py_text_process, fixed_embedding, unit=128):
        super().__init__()
        # Natural language
        self.nl_text_processor = nl_text_processor
        self.nl_voba_size = len(nl_text_processor.get_vocabulary())
        self.nl_embedding = tf.keras.layers.Embedding(
            self.nl_voba_size,
            output_dim=unit,
            mask_zero=True)
        self.fixed_embedding = fixed_embedding
        self.nl_rnn = tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(int(unit/2), return_sequences=True, return_state=True))
        # Attention
        self.attention = tf.keras.layers.Attention()
        # PY
        self.py_text_process = py_text_process
        self.py_voba_size = len(py_text_process.get_vocabulary())
        self.py_embedding = tf.keras.layers.Embedding(
            self.py_voba_size,
            output_dim=unit,
            mask_zero=True)
        self.py_rnn = tf.keras.layers.LSTM(unit, return_sequences=True, return_state=True)
        # Output
        self.out = tf.keras.layers.Dense(self.py_voba_size)
            
    def call(self, nl_text, py_text, training=True):
        nl_tokens = self.nl_text_processor(nl_text) # Shape: (batch, Ts)
        nl_vectors = self.nl_embedding(nl_tokens, training=training) # Shape: (batch, Ts, embedding_dim)
        nl_fixed_vectors = self.fixed_embedding(nl_tokens) # Shape: (batch, Ts, 100)
        nl_combined_vectors = tf.concat([nl_vectors, nl_fixed_vectors], -1) # Shape: (batch, Ts, embedding_dim+100)
        nl_rnn_out, fhstate, fcstate, bhstate, bcstate = self.nl_rnn(nl_vectors, training=training) # Shape: (batch, Ts, bi_rnn_output_dim), (batch, rnn_output_dim) ...
        nl_hstate = tf.concat([fhstate, bhstate], -1)
        nl_cstate = tf.concat([fcstate, bcstate], -1)
        
        py_tokens = self.py_text_process(py_text) # Shape: (batch, Te)
        expected = py_tokens[:,1:] # Shape: (batch, Te-1)
        
        teacher_forcing = py_tokens[:,:-1] # Shape: (batch, Te-1)
        py_vectors = self.py_embedding(teacher_forcing, training=training) # Shape: (batch, Te-1, embedding_dim)
        py_in = self.attention(inputs=[py_vectors,nl_rnn_out], mask=[py_vectors._keras_mask, nl_rnn_out._keras_mask], training=training)
        
        trans_vectors, _, _ = self.py_rnn(py_in, initial_state=[nl_hstate, nl_cstate], training=training) # Shape: (batch, Te-1, rnn_output_dim)
        out = self.out(trans_vectors, training=training) # Shape: (batch, Te-1, py_vocab_size)
        return out, expected, out._keras_mask

with open('./content.json') as content:
  databa = json.load(content)

tags = []
inputs = []
responses={}
for intent in databa['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])

data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})

data = data.sample(frac=1)

import string
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
print(input_shape)

vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
output_length = le.classes_.shape[0]
print("output length: ",output_length)

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model = Model(i, x)

model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

train = model.fit(x_train,y_train,epochs=300)

model1 = BertModel.from_pretrained('bert-base-uncased')
tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')

plt.plot(train.history['accuracy'],label='training set accuracy')
plt.plot(train.history['loss'],label='training set loss')
plt.legend()

while True:
  
  import random
  
  texts_p = []
  prediction_input = input('""" ')
  
  tokens = tokenizer1.tokenize(prediction_input)
  tokens = ['[CLS]'] + tokens + ['[SEP]']
  tokens = tokens + ['[PAD]'] + ['[PAD]']
  attention_mask = [1 if i!= '[PAD]' else 0 for i in tokens]
  token_ids = tokenizer1.convert_tokens_to_ids(tokens)
  token_ids = torch.tensor(token_ids).unsqueeze(0)
  attention_mask = torch.tensor(attention_mask).unsqueeze(0)
  hidden_rep, cls_head = model1(token_ids, attention_mask = attention_mask)

  prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)
  texts_p.append(prediction_input)
 
  prediction_input = tokenizer.texts_to_sequences(texts_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  prediction_input = pad_sequences([prediction_input],input_shape)

 
  output = model.predict(prediction_input)
  output = output.argmax()

  response_tag = le.inverse_transform([output])[0]
  print(random.choice(responses[response_tag]))
  if response_tag == "webserver":
    print("""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Serv(BaseHTTPRequestHandler):

        def do_GET(self):
            if self.path == '/':
                self.path = 'index.html' # your file
            try:
                file_to_open = open(self.path[1:]).read()
                self.send_response(200)

            except:
                file_to_open = "File not find"
                self.send_response(404)
            self.end_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))

    httpd = HTTPServer(('localhost', 8000), Serv)
    httpd.serve_forever()
    """)
  elif response_tag == "fetchtweets":
    print("""
    import tweepy, os 

    def fetch_tweets_from_user(user_name):
        # authentification
        auth = tweepy.OAuthHandler(os.environ['TWITTER_KEY'], os.environ['TWITTER_SECRET'])
        auth.set_access_token(os.environ['TWITTER_TOKEN'], os.environ['TWITTER_TOKEN_SECRET'])
        api = tweepy.API(auth)

        # fetch tweets
        tweets = api.user_timeline(screen_name=user, count=200, include_rts=False)
        return tweets
    """)
  elif response_tag == "workwebsite":
    print("""
    import urllib
    import urllib.request

    try:
        site = urllib.request.urlopen("http://www.dedsecurity.com")
    except urllib.error.URLError:
        print("Error")
    else:
        print("Ok")
        print(site.read())
    """)
