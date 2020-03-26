# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import load_data
import numpy as np

# preprocessing 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# keras for model building

#from keras.models import Model
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

english_sentences = load_data.load('small_vocab_en.txt')
french_sentences = load_data.load('small_vocab_fr.txt')

def tokenize(x):
    x_tkzr = Tokenizer(char_level = False)
    x_tkzr.fit_on_texts(x)
    return x_tkzr.texts_to_sequences(x), x_tkzr

def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

#tests.test_pad(pad)

def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    
# Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
preprocess(english_sentences, french_sentences)
    
    
# convert indexes back to words   
    
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = ' '
    index_to_words[345] = ' '
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)


from keras.models import Sequential

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    learning_rate = 0.005
    
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model


temp_x = pad(preproc_english_sentences, max_french_sequence_length)
model = model_final(temp_x.shape,
                        preproc_french_sentences.shape[1],
                        len(english_tokenizer.word_index)+1,
                        len(french_tokenizer.word_index)+1)
    
model.fit(temp_x, preproc_french_sentences, batch_size = 1024, epochs = 20, validation_split = 0.2)

print(logits_to_text(model.predict(temp_x[:1])[0], french_tokenizer))