#from __future__ import absolute_import, division, print_function, unicode_literals


from helper_functions import Dataset_Helper
from results_saver import LogWriter
from gensim import corpora
import matplotlib.pyplot as plt
from aliaser import *
import os
import sys
import numpy as np





file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
#sess = tf.compat.v1.Session(config=config)
#tf.keras.backend.set_session(sess)
#results_saver = LogWriter(log_file_desc="Autoencoder")
results = []

num_of_words = 10000
dataset_helper = Dataset_Helper(True)
dataset_helper.set_wanted_datasets([2])
dataset_helper.next_dataset()
num_of_topics = dataset_helper.get_num_of_topics()
documents = dataset_helper.get_texts_as_list()
tokenizer = Tokenizer(num_words=num_of_words)
tokenizer.fit_on_texts(documents)
#items= tokenizer.word_index
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
matrix = tokenizer.texts_to_matrix(documents, mode='binary')
#mydict = corpora.Dictionary([line.split() for line in documents],prune_at=num_of_words)
#corpus = [mydict.doc2bow(line.split()) for line in documents]

#tfidf = TfidfModel(corpus)
#print(tfidf)

"""model = Sequential()
model.add(Dense(num_of_words*num_of_topics,activation='relu', input_shape=(num_of_words,)))
model.add(Dense(num_of_words,activation='sigmoid'))"""
input_row = Input(shape=(num_of_words,))
#encoder = Dense(int(num_of_words/2), activation='relu')(input_row)
encoder= Dense(num_of_topics, activation='relu')(input_row)
#decoder = Dense(int(num_of_words/2), activation='relu')(encoder)
output_row = Dense(num_of_words,activation='sigmoid')(encoder)

autoencoder = Model(input_row,output_row)
#autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])#optimizer='adadelta', loss='mse', metrics=['accuracy'])
#autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = autoencoder.fit(matrix,matrix,batch_size=256,epochs=10,validation_split=0.1)
weight_in = autoencoder.get_weights()[0]
weight_out = autoencoder.get_weights()[2]
blob = np.array([])
weight_in = weight_in.transpose()
num_of_important_words = 20
topic_words_in = [[] for i in range(num_of_topics)]
topic_words_out = [[] for i in range(num_of_topics)]
for i in range(num_of_topics):
    lowest_max_in = (-np.sort(-weight_in[i]))[20]
    indexes_in = np.argwhere(weight_in[i]>=lowest_max_in)
    lowest_max_out = (-np.sort(-weight_out[i]))[20]
    indexes_out = np.argwhere(weight_out[i]>=lowest_max_out)
    for index in indexes_in:
        index = index[0]
        topic_words_in[i].append([reverse_word_map[index],weight_in[i,index]])
    for index in indexes_out:
        index = index[0]
        topic_words_out[i].append([reverse_word_map[index],weight_out[i,index]])
from results_saver import LogWriter
saver = LogWriter(log_file_desc='LDATests')
saver.write_2D_list('topic_words_in',topic_words_in)
saver.write_2D_list('topic_words_out',topic_words_out)
#print(topic_words)



