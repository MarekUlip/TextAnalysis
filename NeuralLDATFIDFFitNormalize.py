#from __future__ import absolute_import, division, print_function, unicode_literals
import operator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from NeuralLDAanalysisMethods import *
from helper_functions import Dataset_Helper, stp_wrds
from results_saver import LogWriter
from gensim import corpora
import matplotlib.pyplot as plt
from aliaser import *
import os
import sys
import numpy as np
import gensim
from collections import Counter
from gensim.models import CoherenceModel
from lda_impl import Lda
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from NeuralTopicMatrix import NeuralTopicMatrix, NeuralTopicMatrixTFIDF
import tkinter as tk
from sklearn.preprocessing import normalize
from tkinter import simpledialog

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
root = tk.Tk()
root.withdraw()
norm_to_use = 'l2'
test_name = simpledialog.askstring(title="Test Name",
                                  prompt="Insert test name:",initialvalue='LDATestsFitNormTFIDF{}'.format(norm_to_use))
#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
#sess = tf.compat.v1.Session(config=config)
#tf.keras.backend.set_session(sess)
#results_saver = LogWriter(log_file_desc="Autoencoder")
results = []
#mycolors = np.array([color for name, color in mcolors.XKCD_COLORS.items()])
from sys import getsizeof
num_of_words = 10000
dataset_helper = Dataset_Helper(True)
dataset_helper.set_wanted_datasets([2])
dataset_helper.next_dataset()
num_of_topics = dataset_helper.get_num_of_topics()
documents = dataset_helper.get_texts_as_list()
labels = dataset_helper.get_labels(dataset_helper.get_train_file_path())
#tokenizer = Tokenizer(num_words=num_of_words)
#tokenizer.fit_on_texts(documents)
#items= tokenizer.word_index
#reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
vectorizer = TfidfVectorizer(max_features=num_of_words)
#tst = vectorizer.fit_transform(documents)
#matrix = tokenizer.texts_to_matrix(documents, mode='freq')
matrix = vectorizer.fit_transform(documents).todense()
ftrs = vectorizer.get_feature_names()
reverse_word_map = dictOfWords = {i : ftrs[i] for i in range(0, len(ftrs))}
from sklearn.preprocessing import normalize
#row_sums = matrix.sum(axis=1)
#matrix = matrix / row_sums[:, np.newaxis]
matrix = matrix#normalize(matrix,axis=1,norm='l2')
#mydict = corpora.Dictionary([line.split() for line in documents],prune_at=num_of_words)
#corpus = [mydict.doc2bow(line.split()) for line in documents]

#tfidf = TfidfModel(corpus)
#print(tfidf)

"""model = Sequential()
model.add(Dense(num_of_words*num_of_topics,activation='relu', input_shape=(num_of_words,)))
model.add(Dense(num_of_words,activation='sigmoid'))"""
regularization = 0.001
input_row = Input(shape=(num_of_words,))
#encoder = Dense(int(num_of_words/2), activation='relu')(input_row)
encoder= Dense(num_of_topics, activation='relu',kernel_regularizer=keras.regularizers.l1_l2(regularization))(input_row)#
#encoder= keras.layers.LayerNormalization()(encoder)
#decoder = Dense(int(num_of_words/2), activation='relu')(encoder)
output_row = Dense(num_of_words,activation='softmax',kernel_regularizer=keras.regularizers.l1_l2(regularization))(encoder)

autoencoder = Model(input_row,output_row)
autoencoder.summary()
#autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])#optimizer='adadelta', loss='mse', metrics=['accuracy'])
#autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
"""for i in range(500):
    print('Epoch {}'.format(i))
    autoencoder.ba
    autoencoder.fit(matrix,matrix,batch_size=32,epochs=1,validation_split=0.1, verbose=2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=False)])
    weights = autoencoder.get_weights()
    weights[0] = np.transpose(normalize(np.transpose(weights[0]),norm_to_use,1))
    weights[2] = normalize(weights[2],norm_to_use,1)
    autoencoder.set_weights(weights)"""
batch_size = 32
for i in range(200):
    print('Epoch {}'.format(i))
    for b_ind in range(len(matrix)//batch_size-1):
        history = autoencoder.train_on_batch(matrix[b_ind*batch_size:(b_ind+1)*batch_size],matrix[b_ind*batch_size:(b_ind+1)*batch_size])
        #autoencoder.fit(matrix,matrix,batch_size=32,epochs=1,validation_split=0.1, verbose=2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=False)])
        weights = autoencoder.get_weights()
        #for i in range(len(weights)):
        w_min, w_max = min([weight.min() for weight in weights]),max([weight.max() for weight in weights])
        weights =(weights-w_min)/(w_max-w_min)
        """weights[0] = normalize(weights[0],norm_to_use,0)
        weights[2] = normalize(weights[2],norm_to_use,1)"""
        autoencoder.set_weights(weights)
    print(history)
weight_in = autoencoder.get_weights()[0]
weight_out = autoencoder.get_weights()[2]
#tst = autoencoder.get_weights()
blob = np.array([])

weight_in = weight_in.transpose()
#combined_weight = np.dot(weight_in.transpose(), weight_out)
num_of_important_words = 20

log_writer = LogWriter(log_file_desc='{}{}'.format(test_name,regularization))

log_writer.write_any('model',autoencoder.to_json(),'w+',True)





"""topic_words_in = [sorted(topic_words,key=lambda x: x[1],reverse=True) for topic_words in topic_words_in]
topic_words_out = [sorted(topic_words,key=lambda x: x[1],reverse=True) for topic_words in topic_words_out]
log_writer = LogWriter(log_file_desc='LDATestsRegularize{}'.format(regularization))
log_writer.write_2D_list('topic_words_in', topic_words_in)
log_writer.write_2D_list('topic_words_out', topic_words_out)"""

topic_words_in_max = get_extremes(weight_in,num_of_topics,num_of_important_words,reverse_word_map,True,'topic_words_in_max',log_writer,dataset_helper.get_dataset_name())
topic_words_in_min = get_extremes(weight_in,num_of_topics,num_of_important_words,reverse_word_map,False,'topic_words_in_min',log_writer,dataset_helper.get_dataset_name())
topic_words_out_max = get_extremes(weight_out,num_of_topics,num_of_important_words,reverse_word_map,True,'topic_words_out_max',log_writer,dataset_helper.get_dataset_name())
topic_words_out_min = get_extremes(weight_out,num_of_topics,num_of_important_words,reverse_word_map,False,'topic_words_out_min',log_writer,dataset_helper.get_dataset_name())
#topic_words_combined = get_extremes(combined_weight, num_of_topics, num_of_important_words, reverse_word_map, False, 'topic_words_combined', log_writer, dataset_helper.get_dataset_name())

"""texts = [text.split() for text in documents]
dictionary = corpora.Dictionary(texts)
# TODO maybe test work with dict self.dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]"""
model = Lda(num_of_topics,num_of_important_words,
            passes=4,
            iterations=4)

"""gensim.models.LdaModel(
doc_term_matrix,
num_topics=num_of_topics,
id2word=dictionary,
passes=2,
iterations=2)"""

#LDA section
test_LDA = False
if test_LDA:
    model.train(documents)
    topic_words_lda = extract_important_words(model.get_topics(), True)
    print(topic_words_lda)
    log_writer.write_2D_list('topic_words_lda', topic_words_lda, 'w+')
    test_model(documents, labels, model, log_writer, 'standard_lda')
    plot_clustering_chart(model,True,documents,log_writer,'lda',dataset_helper.get_dataset_name(),dataset_helper.get_num_of_topics())
    measureCoherence(topic_words_lda, log_writer, model.dictionary, documents, 'lda', dataset_helper.get_dataset_name())
else:
    model.dictionary = corpora.Dictionary([text.split() for text in documents])
neural_lda_in = NeuralTopicMatrixTFIDF(weight_in,reverse_word_map,num_of_topics,vectorizer)
neural_lda_out = NeuralTopicMatrixTFIDF(weight_out,reverse_word_map,num_of_topics,vectorizer)
#neural_lda_combined = NeuralTopicMatrix(combined_weight, reverse_word_map,num_of_topics,tokenizer)
test_model(documents, labels, neural_lda_in, log_writer,'neural_lda_in')
test_model(documents, labels, neural_lda_out, log_writer,'neural_lda_out')
#test_model(documents, labels, neural_lda_combined, log_writer,'neural_lda_combined')


measureCoherence(topic_words_in_max,log_writer,model.dictionary,documents,'neural_in_max',dataset_helper.get_dataset_name())
measureCoherence(topic_words_in_min,log_writer,model.dictionary,documents,'neural_in_min',dataset_helper.get_dataset_name())
measureCoherence(topic_words_out_max,log_writer,model.dictionary,documents,'neural_out_max',dataset_helper.get_dataset_name())
measureCoherence(topic_words_out_min,log_writer,model.dictionary,documents,'neural_out_min',dataset_helper.get_dataset_name())
#measureCoherence(topic_words_combined, log_writer, model.dictionary, documents, 'neural_combined', dataset_helper.get_dataset_name())

plot_clustering_chart(neural_lda_out,False,documents,log_writer,'neural_topic_out',dataset_helper.get_dataset_name(),dataset_helper.get_num_of_topics())
plot_clustering_chart(neural_lda_in,False,documents,log_writer,'neural_topic_in',dataset_helper.get_dataset_name(),dataset_helper.get_num_of_topics())
#plot_clustering_chart(neural_lda_combined,False,documents,log_writer,'neural_topic_combined',dataset_helper.get_dataset_name())
log_writer.end_logging()
#plot_clustering_chart(neural_lda_combined,False,documents,log_writer,'neural_topic_combined',dataset_helper.get_dataset_name())

log_writer.end_logging()

#print(topic_words)



