import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Bidirectional, LSTM, Embedding, Flatten
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from training_text_generator_RNN_embedding import Training_Text_Generator_RNN_Embedding
from helper_functions import Dataset_Helper
from results_saver import LogWriter
from embedding_loader import get_embedding_matrix
from keras.utils import plot_model
import os
import sys


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

results_saver = LogWriter(log_file_desc="Autoencoder")
results = []

num_of_words = 1000
num_of_topics = 10

"""model = Sequential()
model.add(Dense(num_of_words*num_of_topics,activation='relu', input_shape=(num_of_words,)))
model.add(Dense(num_of_words,activation='sigmoid'))"""
input_row = Input(shape=(num_of_words,))
hidden1= Dense(num_of_topics * num_of_words, activation='relu')(input_row)
output_row = Dense(num_of_words,activation='sigmoid')(hidden1)

autoencoder = Model(input_row,output_row)
autoencoder.compile(optimizer='rmsprop', loss='mae', metrics=['accuracy'])

#topic_matrix = model.layers[0].weights

plot_model(autoencoder,results_saver.get_plot_path("","model-graph"),show_shapes=True)