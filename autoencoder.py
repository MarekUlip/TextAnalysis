import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Bidirectional, LSTM, Embedding, Flatten
from keras.optimizers import RMSprop, SGD
from keras.utils.np_utils import to_categorical
from training_text_generator_RNN_embedding import Training_Text_Generator_RNN_Embedding
from helper_functions import Dataset_Helper
from results_saver import LogWriter
from embedding_loader import get_embedding_matrix
from keras.utils import plot_model
from gensim import corpora
from gensim.models.tfidfmodel import  TfidfModel
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys


file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

results_saver = LogWriter(log_file_desc="Autoencoder")
results = []

num_of_words = 10000
num_of_topics = 20
dataset_helper = Dataset_Helper(True)
dataset_helper.next_dataset()
documents = dataset_helper.get_texts_as_list()

vectorizer = CountVectorizer(max_features=num_of_words)
matrix = vectorizer.fit_transform(documents).todense()/num_of_words
print(matrix)
#mydict = corpora.Dictionary([line.split() for line in documents],prune_at=num_of_words)
#corpus = [mydict.doc2bow(line.split()) for line in documents]

#tfidf = TfidfModel(corpus)
#print(tfidf)

"""model = Sequential()
model.add(Dense(num_of_words*num_of_topics,activation='relu', input_shape=(num_of_words,)))
model.add(Dense(num_of_words,activation='sigmoid'))"""
input_row = Input(shape=(num_of_words,))
#encoder = Dense(int(num_of_words/2), activation='relu')(input_row)
encoder= Dense(int(num_of_words/num_of_topics), activation='relu')(input_row)
#decoder = Dense(int(num_of_words/2), activation='relu')(encoder)
output_row = Dense(num_of_words,activation='linear')(encoder)

autoencoder = Model(input_row,output_row)
opt = SGD(lr=0.01, momentum=0.9)
autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
#autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
autoencoder.fit(matrix,matrix,batch_size=128,epochs=20,validation_split=0.1)

#topic_matrix = model.layers[0].weights
documents = dataset_helper.get_texts_as_list(dataset_helper.open_file_stream(dataset_helper.get_test_file_path()))

matrix = vectorizer.transform(documents).todense()/num_of_words
count = 0
for row in matrix:
    res = autoencoder.predict(row)
    if np.all([np.round(res,4)*num_of_words, row*num_of_words]):
        count+=1
print("Practical res = {}".format(count/len(documents)))
result = autoencoder.evaluate(matrix,matrix)
print(result)
plot_model(autoencoder,results_saver.get_plot_path("","model-graph"),show_shapes=True)